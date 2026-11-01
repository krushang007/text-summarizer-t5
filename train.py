#!/usr/bin/env python3
"""
SAMSum Dialogue Summarization with T5
Supports Apple Silicon (MPS), CUDA (NVIDIA GPU), and CPU.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Text cleaning helper
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Normalize carriage returns and excessive whitespace
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags if any
    text = re.sub(r"\s+", " ", text)    # collapse multiple spaces
    return text.strip()

def preprocess_function(batch, tokenizer, max_input_length=512, max_target_length=128):
    # Prepend T5 task prefix 'summarize: ' to dialogues
    inputs = ["summarize: " + clean_text(doc) for doc in batch["dialogue"]]
    targets = [clean_text(doc) for doc in batch["summary"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def build_compute_metrics(tokenizer, metric):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Replace -100 in preds and labels so tokenizers can decode safely
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}
    return compute_metrics

def summarize_dialogue(dialogue: str, model, tokenizer, device, max_length=128, num_beams=4):
    """Run inference to generate a summary for a dialogue."""
    model.eval()
    cleaned = clean_text(dialogue)
    prompt = "summarize: " + cleaned

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Train T5 on SAMSum dataset")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Model checkpoint (t5-small, t5-base, etc.)")
    parser.add_argument("--train_file", type=str, default="samsum-train.csv")
    parser.add_argument("--val_file", type=str, default="samsum-validation.csv")
    parser.add_argument("--test_file", type=str, default="samsum-test.csv")
    parser.add_argument("--output_dir", type=str, default="./saved_summary_model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit number of train samples for quick training")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit number of val samples")
    parser.add_argument("--inference_only", action="store_true", help="Skip training and run inference using saved model")
    args = parser.parse_args()

    # Determine optimal compute device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU via Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    sample_dialogue = """
Reporter: In today's technology news, artificial intelligence continues to expand rapidly across industries, from healthcare to finance and education. Recent reports suggest that AI adoption has significantly increased over the past few years.
Reporter: Companies are investing heavily in machine learning systems to automate tasks, improve decision-making, and enhance customer experiences. However, this growth has also raised questions about job displacement and ethical concerns.
Expert: AI systems are becoming more capable due to advances in deep learning and access to large datasets. These models can now perform complex tasks such as language understanding, image recognition, and even code generation.
Expert: At the same time, there are valid concerns about bias in AI models, as they often reflect the data they are trained on. Ensuring fairness and transparency is becoming a key area of research.
Reporter: Governments and organizations are beginning to introduce regulations to guide the development and deployment of AI technologies. The goal is to balance innovation with accountability.
"""

    if args.inference_only:
        print(f"Loading fine-tuned model and tokenizer from '{args.output_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir).to(device)
        print("\n--- Running Inference on Sample Dialogue ---")
        summary = summarize_dialogue(sample_dialogue, model, tokenizer, device)
        print("\nGenerated Summary:\n", summary)
        return

    # Load data
    print(f"Loading CSV datasets from {args.train_file}, {args.val_file}...")
    train_df = pd.read_csv(args.train_file).dropna(subset=["dialogue", "summary"])
    val_df = pd.read_csv(args.val_file).dropna(subset=["dialogue", "summary"])

    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = train_df.sample(n=args.max_train_samples, random_state=42).reset_index(drop=True)
    if args.max_val_samples and args.max_val_samples < len(val_df):
        val_df = val_df.sample(n=args.max_val_samples, random_state=42).reset_index(drop=True)

    print(f"Training dataset size: {len(train_df)} samples")
    print(f"Validation dataset size: {len(val_df)} samples")

    # Load tokenizer and base model
    print(f"Loading pretrained model and tokenizer: '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(
        lambda b: preprocess_function(b, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        lambda b: preprocess_function(b, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge_metric = evaluate.load("rouge")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        logging_steps=20,
        logging_first_step=True,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer, rouge_metric),
    )

    print("Starting training...")
    trainer.train()

    print(f"\nSaving final model and tokenizer to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model saved successfully!")

    # Test inference with the saved model
    print("\nTesting inference with the trained model...")
    model.to(device)
    summary = summarize_dialogue(sample_dialogue, model, tokenizer, device)
    print("\n--- SAMPLE DIALOGUE ---")
    print(sample_dialogue.strip())
    print("\n--- GENERATED SUMMARY ---")
    print(summary)

if __name__ == "__main__":
    main()

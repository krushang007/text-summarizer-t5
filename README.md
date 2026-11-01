# Dialogue Text Summarizer with T5

Fine-tuned Google T5 (`t5-small`) for dialogue summarization on the SAMSum dataset.

Supports:
- **Apple Silicon (MPS)** hardware acceleration
- **NVIDIA GPU (CUDA)**
- **CPU**
- **Google Colab**

## Project Structure

```text
├── saved_summary_model/             # Pre-trained model weights (Git LFS) & tokenizer (Ready to run!)
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── textSummarizer (2).ipynb         # Interactive Jupyter notebook (Colab & local ready)
├── train.py                         # Standalone CLI training & inference script
├── samsum-train.csv                 # SAMSum training dataset
├── samsum-validation.csv            # SAMSum validation dataset
├── samsum-test.csv                  # SAMSum test dataset
├── requirements.txt                 # Python dependencies
├── .gitattributes                   # Git LFS tracking configuration
└── .gitignore                       # Clean repository ignore rules
```

## Running on Another PC (No Retraining Needed!)

The trained model is already included in the repository via Git LFS.

### 1. Clone the Repository
```bash
git clone https://github.com/krushang007/text-summarizer-t5.git
cd text-summarizer-t5
git lfs pull
```

### 2. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Inference Immediately
```bash
python train.py --inference_only
```

Or load the model directly in Python:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("./saved_summary_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./saved_summary_model")

dialogue = "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)"
inputs = tokenizer("summarize: " + dialogue, return_tensors="pt").to(device)
summary_ids = model.generate(**inputs, max_new_tokens=64, num_beams=4)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

---

## (Optional) Retraining the Model

If you ever wish to re-train the model:
```bash
python train.py --model_name t5-small --epochs 4 --batch_size 4 --grad_accum 2
```

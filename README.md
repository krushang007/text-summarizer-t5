# Dialogue Text Summarizer with T5

Fine-tuning Google T5 (`t5-small` / `t5-base`) for dialogue summarization on the SAMSum dataset.

Supports:
- **Apple Silicon (MPS)** hardware acceleration
- **NVIDIA GPU (CUDA)**
- **CPU**
- **Google Colab**

## Project Structure

```text
├── textSummarizer (2).ipynb         # Interactive Jupyter notebook (Colab & local ready)
├── train.py                         # Standalone CLI training & inference script
├── samsum-train.csv                 # SAMSum training dataset
├── samsum-validation.csv            # SAMSum validation dataset
├── samsum-test.csv                  # SAMSum test dataset
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Ignored artifacts & large model files
```

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Model
```bash
# Fast training run (e.g. 4,000 samples, ~10-15 mins on Apple Silicon M-series)
python train.py --model_name t5-small --epochs 3 --batch_size 4 --grad_accum 2 --max_train_samples 4000

# Full dataset training (all 14,732 samples)
python train.py --model_name t5-small --epochs 3 --batch_size 4 --grad_accum 2
```

### 3. Run Inference on Custom Text
```bash
python train.py --inference_only
```

Or open and run `textSummarizer (2).ipynb` in VS Code / Jupyter / Google Colab.

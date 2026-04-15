#!/bin/bash
# ==============================================================
# setup_nscc.sh — Run ONCE interactively before submitting jobs.
# Sets up the Python environment and downloads data/models.
#
# Usage (on login node):
#   cd /home/users/$USER/Text-Emotion-Recognition
#   bash jobs/setup_nscc.sh
# ==============================================================

set -e

echo "=========================================="
echo "NSCC Setup: Text Emotion Recognition"
echo "=========================================="

# --- 1. Create virtual environment ---
echo "[1/5] Creating Python virtual environment..."
module load python/3.10
python -m venv ~/ter_env
source ~/ter_env/bin/activate

# --- 2. Install dependencies ---
echo "[2/5] Installing Python packages..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas scikit-learn matplotlib seaborn tqdm requests

# --- 3. Download BERT model (cache it so jobs don't download at runtime) ---
echo "[3/5] Pre-downloading BERT model..."
python -c "
from transformers import AutoModel, AutoTokenizer
for model_name in ['bert-base-uncased']:
    print(f'  Downloading {model_name}...')
    AutoModel.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(model_name)
print('  Done.')
"

# --- 4. Prepare data directories ---
echo "[4/5] Setting up data directories..."
mkdir -p data/wassa2017

# Check if datasets exist
if [ ! -f data/crowdflower.csv ]; then
    echo "  ⚠ data/crowdflower.csv not found!"
    echo "    Please upload it from Kaggle:"
    echo "    https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text"
fi

# Download WASSA2017 if not present
WASSA_COUNT=$(ls data/wassa2017/*.txt 2>/dev/null | wc -l)
if [ "$WASSA_COUNT" -lt 4 ]; then
    echo "  Downloading WASSA2017 dataset..."
    git clone --depth 1 https://github.com/vinayakumarr/WASSA-2017.git /tmp/wassa_repo 2>/dev/null || true
    cp /tmp/wassa_repo/wassa/data/training/* data/wassa2017/ 2>/dev/null || true
    cp /tmp/wassa_repo/wassa/data/testing/* data/wassa2017/ 2>/dev/null || true
    cp /tmp/wassa_repo/wassa/data/validation/* data/wassa2017/ 2>/dev/null || true
    rm -rf /tmp/wassa_repo
    echo "  WASSA2017 downloaded."
else
    echo "  WASSA2017 already present ($WASSA_COUNT files)."
fi

# GloVe (optional — for TextCNN/BiLSTM baselines)
if [ ! -f data/glove.6B.300d.txt ]; then
    echo "  Downloading GloVe embeddings (822MB)..."
    wget -q https://nlp.stanford.edu/data/glove.6B.zip -O /tmp/glove.6B.zip
    unzip -q -o /tmp/glove.6B.zip glove.6B.300d.txt -d data/
    rm /tmp/glove.6B.zip
    echo "  GloVe downloaded."
else
    echo "  GloVe already present."
fi

# --- 5. Create output directories ---
echo "[5/5] Creating output directories..."
mkdir -p outputs figures checkpoints logs

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Verify: ls data/crowdflower.csv data/wassa2017/*.txt data/glove.6B.300d.txt"
echo "  2. Submit jobs:"
echo "     qsub jobs/job_step1.pbs"
echo "     qsub jobs/job_step2.pbs"
echo "     qsub jobs/job_step3.pbs"
echo "     qsub jobs/job_step3b.pbs"
echo "     # After all complete:"
echo "     qsub jobs/job_step4.pbs"
echo "=========================================="

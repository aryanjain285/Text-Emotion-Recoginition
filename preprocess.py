"""
preprocess.py — Data acquisition, cleaning, tokenization, and DataLoader creation.

Handles both CROWDFLOWER and WASSA2017 datasets.
Provides a unified interface so all models receive data in the same format.

Usage:
    from preprocess import load_data
    datasets, label_names, num_classes = load_data(cfg)
"""

import os
import re
import csv
import json
import logging
import requests
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("project")


# ============================================================================
#  1. Dataset Download & Loading
# ============================================================================

def download_file(url: str, dest: str) -> None:
    """Download a file from URL with progress indication."""
    if os.path.exists(dest):
        logger.info(f"File already exists: {dest}")
        return
    logger.info(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
    print()


def load_crowdflower(data_dir: str, top_k: int = 6) -> pd.DataFrame:
    """
    Load the CROWDFLOWER Sentiment Analysis in Text dataset.

    The dataset contains ~40K tweets labeled with 13 emotion categories.
    We keep only the top-K most frequent emotions for cleaner experiments,
    as rare classes have very few samples and add noise.

    The CSV is expected at: {data_dir}/crowdflower.csv
    Columns: tweet_id, sentiment, content

    If not found, provides download instructions.

    Args:
        data_dir: Directory containing the dataset.
        top_k: Number of top emotion classes to keep.

    Returns:
        DataFrame with columns ['text', 'emotion'].
    """
    csv_path = os.path.join(data_dir, "crowdflower.csv")

    # Try alternative filenames the user might have
    alt_names = [
        "text_emotion.csv", "sentiment_analysis.csv",
        "crowdflower_data.csv", "tweet_emotions.csv"
    ]
    if not os.path.exists(csv_path):
        for alt in alt_names:
            alt_path = os.path.join(data_dir, alt)
            if os.path.exists(alt_path):
                csv_path = alt_path
                break

    if not os.path.exists(csv_path):
        msg = (
            f"CROWDFLOWER dataset not found in {data_dir}.\n"
            "Please download it from:\n"
            "  https://data.world/crowdflower/sentiment-analysis-in-text\n"
            "  or https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text\n\n"
            f"Save the CSV file as: {os.path.join(data_dir, 'crowdflower.csv')}\n"
            "Expected columns: tweet_id, sentiment, content"
        )
        raise FileNotFoundError(msg)

    # Read CSV - handle potential encoding issues with crowdsourced data
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")

    # Normalize column names (different sources use different names)
    df.columns = df.columns.str.strip().str.lower()

    # Map to standard column names
    text_col = None
    emotion_col = None
    for col in df.columns:
        if col in ["content", "text", "tweet", "sentence"]:
            text_col = col
        if col in ["sentiment", "emotion", "label", "feeling"]:
            emotion_col = col

    if text_col is None or emotion_col is None:
        raise ValueError(
            f"Could not identify text/emotion columns. Found: {list(df.columns)}\n"
            "Expected columns like 'content'/'text' and 'sentiment'/'emotion'."
        )

    df = df[[text_col, emotion_col]].rename(
        columns={text_col: "text", emotion_col: "emotion"}
    )

    # Clean up emotion labels
    df["emotion"] = df["emotion"].str.strip().str.lower()
    df = df.dropna(subset=["text", "emotion"])
    df = df[df["text"].str.len() > 0]

    # Keep only top-K most frequent emotions
    emotion_counts = df["emotion"].value_counts()
    top_emotions = emotion_counts.head(top_k).index.tolist()
    df = df[df["emotion"].isin(top_emotions)].reset_index(drop=True)

    logger.info(f"CROWDFLOWER: {len(df)} samples, {df['emotion'].nunique()} classes")
    logger.info(f"  Emotion distribution:\n{df['emotion'].value_counts().to_string()}")

    return df


def load_wassa2017(data_dir: str) -> pd.DataFrame:
    """
    Load the WASSA2017 Shared Task dataset.

    4 emotions: anger, fear, joy, sadness.
    ~7K samples from tweets with emotion hashtags.

    The data is expected at: {data_dir}/wassa2017/
    with train/dev/test files, OR as combined CSV.

    If not found, attempts to download from GitHub.

    Returns:
        DataFrame with columns ['text', 'emotion'].
    """
    wassa_dir = os.path.join(data_dir, "wassa2017")
    os.makedirs(wassa_dir, exist_ok=True)

    combined_path = os.path.join(wassa_dir, "wassa_combined.csv")

    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        logger.info(f"WASSA2017: Loaded from cached combined CSV, {len(df)} samples")
        return df

    # Try to download from the GitHub repository
    base_url = "https://raw.githubusercontent.com/vinayakumarr/WASSA-2017/master/wassa/"
    emotions = ["anger", "fear", "joy", "sadness"]
    splits = ["train", "dev", "test"]

    all_rows = []
    downloaded_any = False

    for emotion in emotions:
        for split in splits:
            # Try multiple filename patterns used in the repo
            patterns = [
                f"{emotion}-ratings-0to1.{split}.txt",
                f"{emotion}/{split}.txt",
                f"{split}/{emotion}.txt",
                f"{emotion}-{split}.txt",
            ]

            for pattern in patterns:
                url = base_url + pattern
                local_path = os.path.join(wassa_dir, f"{emotion}_{split}.txt")

                if os.path.exists(local_path):
                    # Already downloaded
                    try:
                        with open(local_path, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split("\t")
                                if len(parts) >= 2:
                                    text = parts[1] if len(parts) == 3 else parts[0]
                                    all_rows.append({"text": text, "emotion": emotion})
                                    downloaded_any = True
                    except Exception:
                        pass
                    break

                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200 and len(resp.text) > 50:
                        with open(local_path, "w", encoding="utf-8") as f:
                            f.write(resp.text)

                        for line in resp.text.strip().split("\n"):
                            parts = line.strip().split("\t")
                            if len(parts) >= 2:
                                # Format: id \t text \t score  OR  text \t score
                                text = parts[1] if len(parts) >= 3 else parts[0]
                                all_rows.append({"text": text, "emotion": emotion})
                                downloaded_any = True
                        break
                except Exception:
                    continue

    if not downloaded_any:
        # Provide clear fallback instructions
        # Also try to find any CSV/TSV files the user may have placed
        for fname in os.listdir(wassa_dir):
            fpath = os.path.join(wassa_dir, fname)
            if fname.endswith((".csv", ".tsv", ".txt")):
                try:
                    sep = "\t" if fname.endswith(".tsv") or fname.endswith(".txt") else ","
                    tmp = pd.read_csv(fpath, sep=sep, header=None, on_bad_lines="skip")
                    if len(tmp.columns) >= 2:
                        for _, row in tmp.iterrows():
                            vals = [str(v).strip() for v in row if pd.notna(v)]
                            # Heuristic: longest field is text, short one is emotion
                            if len(vals) >= 2:
                                text = max(vals, key=len)
                                label = min(vals, key=len)
                                if label.lower() in emotions:
                                    all_rows.append({"text": text, "emotion": label.lower()})
                except Exception:
                    continue

    if not all_rows:
        msg = (
            f"WASSA2017 dataset not found or empty in {wassa_dir}.\n"
            "Please download it from:\n"
            "  https://github.com/vinayakumarr/WASSA-2017/tree/master/wassa\n\n"
            "Option 1: Place the raw text files in the wassa2017/ folder.\n"
            "Option 2: Create a CSV with columns 'text' and 'emotion' (anger/fear/joy/sadness)\n"
            f"  and save as: {combined_path}"
        )
        raise FileNotFoundError(msg)

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["text", "emotion"])
    df = df[df["text"].str.len() > 5]  # Filter very short artifacts
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["emotion"] = df["emotion"].str.strip().str.lower()

    # Cache for next time
    df.to_csv(combined_path, index=False)

    logger.info(f"WASSA2017: {len(df)} samples, {df['emotion'].nunique()} classes")
    logger.info(f"  Emotion distribution:\n{df['emotion'].value_counts().to_string()}")

    return df


# ============================================================================
#  2. Text Cleaning
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean tweet-style text for emotion classification.

    Cleaning steps:
    1. Lowercase
    2. Remove URLs
    3. Remove @mentions (but preserve the context around them)
    4. Remove hashtag symbols (keep the word: #happy -> happy)
    5. Remove excessive punctuation and special characters
    6. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)           # URLs
    text = re.sub(r"@\w+", "", text)                         # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                    # #hashtags -> word
    text = re.sub(r"[^a-z0-9\s.,!?'\"-]", " ", text)        # Special chars
    text = re.sub(r"([!?.])\1+", r"\1", text)                # Repeated punct
    text = re.sub(r"\s+", " ", text).strip()                 # Collapse spaces

    return text


# ============================================================================
#  3. Tokenized Dataset Classes
# ============================================================================

class EmotionDataset(Dataset):
    """
    PyTorch Dataset for BERT-based models.

    Tokenizes text using a HuggingFace tokenizer and returns
    input_ids, attention_mask, and label tensors.

    Args:
        texts: List of cleaned text strings.
        labels: List of integer labels.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SimpleTextDataset(Dataset):
    """
    Dataset for non-BERT models (TextCNN, BiLSTM) that use
    integer-indexed vocabularies instead of subword tokenizers.

    Args:
        texts: List of cleaned text strings.
        labels: List of integer labels.
        vocab: Dictionary mapping word -> index.
        max_length: Maximum sequence length (pad/truncate).
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.pad_idx = vocab.get("<PAD>", 0)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        words = self.texts[idx].split()[:self.max_length]
        ids = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]

        # Pad to max_length
        padding = [self.pad_idx] * (self.max_length - len(ids))
        ids = ids + padding
        mask = [1] * min(len(words), self.max_length) + [0] * len(padding)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================================
#  4. Vocabulary Builder (for non-BERT models)
# ============================================================================

def build_vocab(
    texts: List[str],
    max_vocab_size: int = 25000,
    min_freq: int = 2
) -> Dict[str, int]:
    """
    Build a word-level vocabulary from training texts.

    Special tokens:
        <PAD> = 0, <UNK> = 1

    Args:
        texts: Training text strings.
        max_vocab_size: Maximum vocabulary size.
        min_freq: Minimum word frequency to include.

    Returns:
        Dictionary mapping word -> integer index.
    """
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in word_counts.most_common(max_vocab_size):
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    logger.info(f"Vocabulary size: {len(vocab)}")
    return vocab


def load_glove_embeddings(
    vocab: Dict[str, int],
    glove_path: str,
    embedding_dim: int = 300
) -> torch.Tensor:
    """
    Load pretrained GloVe embeddings for the vocabulary.

    If GloVe file not found, returns random initialization
    with a warning (the model will still work, just without pretrained embeds).

    Args:
        vocab: Word -> index mapping.
        glove_path: Path to GloVe .txt file (e.g., glove.6B.300d.txt).
        embedding_dim: Embedding dimension (must match GloVe file).

    Returns:
        Embedding weight matrix of shape [vocab_size, embedding_dim].
    """
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embeddings[0] = 0  # <PAD> should be zero vector

    if not os.path.exists(glove_path):
        logger.warning(
            f"GloVe file not found: {glove_path}\n"
            "Using random initialization. For better baseline results, download:\n"
            "  https://nlp.stanford.edu/data/glove.6B.zip\n"
            "  and extract glove.6B.300d.txt to the data directory."
        )
        return torch.FloatTensor(embeddings)

    loaded = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == embedding_dim:
                    embeddings[vocab[word]] = vec
                    loaded += 1

    coverage = loaded / (len(vocab) - 2) * 100  # Exclude PAD and UNK
    logger.info(f"GloVe: loaded {loaded}/{len(vocab)-2} words ({coverage:.1f}% coverage)")

    return torch.FloatTensor(embeddings)


# ============================================================================
#  5. Master Data Loading Function
# ============================================================================

def load_data(cfg) -> Dict:
    """
    Master function: load, clean, split, tokenize, and create DataLoaders.

    This is the single entry point for all data needs. Returns a dictionary
    containing everything the training pipeline needs.

    Args:
        cfg: ProjectConfig object.

    Returns:
        Dictionary with structure:
        {
            "crowdflower": {  # (or "wassa2017")
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "test_loader": DataLoader,
                "train_loader_simple": DataLoader,  # For non-BERT models
                "val_loader_simple": DataLoader,
                "test_loader_simple": DataLoader,
                "label_names": List[str],
                "num_classes": int,
                "class_weights": Tensor,
                "vocab": Dict,             # For non-BERT models
                "glove_embeddings": Tensor, # For non-BERT models
            },
            ...
        }
    """
    data_cfg = cfg.data
    path_cfg = cfg.paths
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)

    result = {}

    # Determine which datasets to load
    datasets_to_load = []
    if data_cfg.dataset in ["crowdflower", "both"]:
        datasets_to_load.append("crowdflower")
    if data_cfg.dataset in ["wassa2017", "both"]:
        datasets_to_load.append("wassa2017")

    for ds_name in datasets_to_load:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading dataset: {ds_name}")
        logger.info(f"{'='*60}")

        # ----- Load raw data -----
        if ds_name == "crowdflower":
            df = load_crowdflower(path_cfg.data_dir, top_k=data_cfg.crowdflower_top_k_emotions)
        else:
            df = load_wassa2017(path_cfg.data_dir)

        # ----- Clean text -----
        df["text"] = df["text"].apply(clean_text)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)

        # ----- Encode labels -----
        label_names = sorted(df["emotion"].unique().tolist())
        label2idx = {name: idx for idx, name in enumerate(label_names)}
        df["label"] = df["emotion"].map(label2idx)
        num_classes = len(label_names)

        logger.info(f"Classes ({num_classes}): {label_names}")
        logger.info(f"Label mapping: {label2idx}")

        # ----- Stratified split -----
        texts = df["text"].tolist()
        labels = df["label"].tolist()

        # First split: train+val vs test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=data_cfg.test_ratio,
            stratify=labels,
            random_state=42
        )

        # Second split: train vs val
        val_frac = data_cfg.val_ratio / (1.0 - data_cfg.test_ratio)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_frac,
            stratify=train_val_labels,
            random_state=42
        )

        logger.info(f"Split sizes - Train: {len(train_texts)}, "
                     f"Val: {len(val_texts)}, Test: {len(test_texts)}")

        # ----- Compute class weights for imbalanced data -----
        label_counts = Counter(train_labels)
        total = sum(label_counts.values())
        class_weights = torch.tensor(
            [total / (num_classes * label_counts[i]) for i in range(num_classes)],
            dtype=torch.float32
        )
        logger.info(f"Class weights: {class_weights.tolist()}")

        # ----- Create BERT datasets -----
        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, data_cfg.max_seq_length)
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, data_cfg.max_seq_length)
        test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, data_cfg.max_seq_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # ----- Create simple (non-BERT) datasets -----
        vocab = build_vocab(train_texts)
        glove_path = os.path.join(path_cfg.data_dir, "glove.6B.300d.txt")
        glove_embeddings = load_glove_embeddings(
            vocab, glove_path, embedding_dim=cfg.model.embedding_dim
        )

        train_simple = SimpleTextDataset(train_texts, train_labels, vocab, data_cfg.max_seq_length)
        val_simple = SimpleTextDataset(val_texts, val_labels, vocab, data_cfg.max_seq_length)
        test_simple = SimpleTextDataset(test_texts, test_labels, vocab, data_cfg.max_seq_length)

        train_loader_simple = DataLoader(
            train_simple, batch_size=cfg.train.batch_size,
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader_simple = DataLoader(
            val_simple, batch_size=cfg.train.batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        test_loader_simple = DataLoader(
            test_simple, batch_size=cfg.train.batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )

        result[ds_name] = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "train_loader_simple": train_loader_simple,
            "val_loader_simple": val_loader_simple,
            "test_loader_simple": test_loader_simple,
            "label_names": label_names,
            "num_classes": num_classes,
            "class_weights": class_weights,
            "vocab": vocab,
            "glove_embeddings": glove_embeddings,
            # Store raw splits for error analysis later
            "test_texts": test_texts,
            "test_labels": test_labels,
        }

    return result

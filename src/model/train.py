"""
train.py — Fine-tune DistilBERT for toxic content classification.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/processed/train.csv"
EVAL_PATH = "data/processed/val.csv"
OUTPUT_DIR = "models/finetuned"
BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_LEN = 128
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ───────────────────────────────────────────────────────────────────
class ToxicCommentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    pred_classes = np.argmax(preds, axis=1)
    precision = precision_score(labels, pred_classes, zero_division=0)
    recall = recall_score(labels, pred_classes, zero_division=0)
    f1 = f1_score(labels, pred_classes, zero_division=0)
    logger.info("\n" + classification_report(labels, pred_classes, target_names=["clean", "toxic"]))
    return {"precision": precision, "recall": recall, "f1": f1}


# ─── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, val_loader) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            all_preds.append(logits)
            all_labels.append(labels)

    return compute_metrics(np.vstack(all_preds), np.concatenate(all_labels))


# ─── Save ──────────────────────────────────────────────────────────────────────
def save_model(model, tokenizer):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")


# ─── Training Loop ─────────────────────────────────────────────────────────────
def train():
    logger.info(f"Training on: {DEVICE}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    train_df = pd.read_csv(DATA_PATH)
    val_df = pd.read_csv(EVAL_PATH)
    logger.info(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

    train_dataset = ToxicCommentDataset(train_df, tokenizer)
    val_dataset = ToxicCommentDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        metrics = evaluate(model, val_loader)
        logger.info(f"Validation Metrics: {metrics}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_model(model, tokenizer)
            logger.info(f"Best model saved (F1={best_f1:.4f})")

    logger.info(f"Training complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/train.csv"
OUTPUT_DIR = "data/processed"

def preprocess():
    logger.info(f"Loading data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    logger.info(f"Dataset shape: {df.shape}")

    # Our synthetic data already has text and label columns
    df = df[["text", "label"]].dropna()

    dist = df["label"].value_counts()
    logger.info(f"Class distribution:\n{dist}")

    # Split 80/10/10
    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    logger.info(f"Saved -> train: {len(train)} | val: {len(val)} | test: {len(test)}")

if __name__ == "__main__":
    preprocess()

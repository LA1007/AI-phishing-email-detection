import numpy as np
import pandas as pd
import torch
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def validate_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Email Text", "Email Type"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Expected columns: {required_columns}"
        )

    df = df[["Email Text", "Email Type"]].copy()
    df = df.dropna(subset=["Email Text", "Email Type"])

    df["Email Type"] = df["Email Type"].astype(str).str.strip().str.lower()
    df["Email Text"] = df["Email Text"].astype(str).str.strip()

    df = df[
        (df["Email Text"].str.len() > 0) &
        (df["Email Text"].str.lower() != "empty")
    ].copy()

    label_map = {
        "phishing email": 1,
        "safe email": 0,
    }

    unknown_labels = sorted(set(df["Email Type"]) - set(label_map.keys()))
    if unknown_labels:
        raise ValueError(
            f"Found unexpected label values in 'Email Type': {unknown_labels}. "
            "Allowed values are: 'Phishing' and 'Safe'."
        )

    df["label"] = df["Email Type"].map(label_map)
    df = df[df["Email Text"].str.len() < 50000].reset_index(drop=True)

    return df


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def stratified_split(df: pd.DataFrame, random_state):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def print_split_stats(name: str, split_df: pd.DataFrame) -> None:
    counts = split_df["label"].value_counts().to_dict()
    safe_count = counts.get(0, 0)
    phishing_count = counts.get(1, 0)
    print(f"{name}: {len(split_df)} samples | Safe: {safe_count} | Phishing: {phishing_count}")

def clean_csv(path: str, email_type: str):
    with open(path, 'r') as file:
        raw_text = file.read()
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    
    fixed_rows = ["Email Text,Email Type"]
    for line in lines[1:]:
        text, label = line.rsplit(",", 1)
        text = '"' + text.replace('"', '""') + '"'
        fixed_rows.append(f"{text},{email_type}")
    
    fixed_csv = "\n".join(fixed_rows)
    return pd.read_csv(StringIO(fixed_csv))
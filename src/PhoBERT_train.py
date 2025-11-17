import pandas as pd
import os
import numpy as np
import torch 
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LRScheduler
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from py_vncorenlp import VnCoreNLP
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
from utils import (
    remove_repetitive_characters,
    remove_stopwords,
    correct_spelling_teencode,
    standardize_data,
    load_vncorenlp,
    EarlyStopping
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data", "raw", "labeled_data.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")

def preprocess_data(input_path, max_length=128, force_preprocess=False):
    """
    Preprocess data specifically for PhoBERT model
    
    Args:
        input_path: Path to the input CSV file
        output_dir: Directory to save processed data (if None, don't save)
        max_length: Maximum sequence length for tokenization
        force_preprocess: Whether to force preprocessing even if processed data exists
    
    Returns:
        Processed training and testing data (tensor)
    """
    if not force_preprocess:
        try: 
            train_input_ids = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_train_input_ids.npy"))
            train_attention_masks = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_train_attention_masks.npy"))
            train_labels = np.load(os.path.join(PROCESSED_DATA_DIR,"phobert_train_labels.npy"))
            test_input_ids = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_test_input_ids.npy"))
            test_attention_masks = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_test_attention_masks.npy"))
            test_labels = np.load(os.path.join(PROCESSED_DATA_DIR, "phobert_test_labels.npy"))
            print("Loaded preprocessed data from disk.")
            return (train_input_ids, train_attention_masks, train_labels), (test_input_ids, test_attention_masks, test_labels)
        except (FileNotFoundError, IOError):
            print("Preprocessed data not found. Performing preprocessing...")

        

    # Load data
    df = pd.read_csv(input_path)
    df = df[["comment", "label"]].dropna()

    # Mapping labels: NEG=0, NEU=1, POS=2
    label_mapping = {"NEG": 0, "NEU": 1, "POS": 2}
    df["label"] = df["label"].map(label_mapping)


    # Preprocessing
    print("Applying Vietnamese text preprocessing...")
    vn_segmenter = load_vncorenlp()
    df["processed_comment"] = (
        df["comment"]
        .astype(str) 
        .apply(standardize_data)
        .apply(remove_repetitive_characters)
        .apply(lambda x: correct_spelling_teencode(x, {}, vn_segmenter))
    )
    df = df[df["processed_comment"].str.strip() != ""]

    # Initialize PhoBERT Tokenizer
    print("Loading PhoBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    def tokenize_text(text):
        try:
            segmented_text = ' '.join([' '.join(sent) for sent in vn_segmenter.word_segment(text)])
        except Exception as e:
            segmented_text = text 

        encoded = tokenizer(
            segmented_text,
            padding = "max_length",
            truncation = True,
            max_length = max_length,
            return_tensors = "pt"
            )
        return {
            'input_ids': encoded['input_ids'].squeeze().numpy(),
            'attention_mask': encoded['attention_mask'].squeeze().numpy()
        }

    print("Tokenizing data...")
    # Apply tokenization
    tokenized_data = [tokenize_text(text) for text in tqdm(df["processed_comment"], desc="Tokenizing")]

    # Extract input_ids and attention_mask
    input_ids = np.array([t["input_ids"] for t in tokenized_data])
    attention_masks = np.array([t["attention_mask"] for t in tokenized_data])
    labels = df["label"].values

    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, mask_train, mask_test, y_train, y_test = train_test_split(
        input_ids, attention_masks, labels, test_size=0.2, stratify=labels, random_state=42
    )


    # Save processed data if output_dir is provided
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_train_input_ids.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_train_attention_masks.npy"), mask_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_train_labels.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_test_input_ids.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_test_attention_masks.npy"), mask_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "phobert_test_labels.npy"), y_test)

    print("Preprocessing completed!")
    return (X_train, mask_train, y_train), (X_test, mask_test, y_test)
    
    # Create PyTorch DataLoaders for training and testing

def create_dataloaders(train_data, test_data, batch_size):
    def to_tensor(data):
        return tuple(torch.tensor(d, dtype=torch.long) for d in data)
    train_tensors = to_tensor(train_data)
    test_tensors = to_tensor(test_data)
    train_ds = TensorDataset(*train_tensors)
    test_ds = TensorDataset(*test_tensors)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size)
    )
    

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss, preds, labels_all = 0, [], []
    for batch in tqdm(dataloader, desc=f"Epoch {epoch +1} - training"):
        ids, mask, y = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=ids, attention_mask=mask, labels=y)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    precision = precision_score(labels_all, preds, average="macro", zero_division=0)
    recall = recall_score(labels_all, preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / len(dataloader),
        "acc": accuracy_score(labels_all, preds),
        "f1": f1_score(labels_all, preds, average="macro"),
        "precision": precision,
        "recall": recall
    }

def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_loss, preds, labels_all = 0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} - Evaluating"):
            ids, mask, y = [b.to(device) for b in batch]
            outputs = model(input_ids=ids, attention_mask=mask, labels=y)
            total_loss += outputs.loss.item()
            preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    precision = precision_score(labels_all, preds, average="macro", zero_division=0)
    recall = recall_score(labels_all, preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / len(dataloader),
        "acc": accuracy_score(labels_all, preds),
        "f1": f1_score(labels_all, preds, average="macro"),
        "precision": precision,
        "recall": recall
    }


def main():
    config_path = os.path.join(BASE_DIR, "configs", "phobert.yml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["model_variants"]["phobert_tuned_1"]

    batch_size, epochs = cfg["batch_size"], cfg["epochs"]
    lr, patience = float(cfg["learning_rate"]), cfg["patience"]
    warmup_ratio, max_length = cfg["warmup_steps_ratio"], cfg["max_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, test_data = preprocess_data(DATA_DIR, max_length=max_length)
    train_loader, test_loader = create_dataloaders(train_data, test_data, batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/phobert-base", num_labels=3
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model_path = os.path.join(BASE_DIR, "models", "phobert_sentiment_classifier.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    mlflow.set_experiment("vietnamese_sentiment_phobert")
    with mlflow.start_run(run_name="phobert_sentiment"):
        mlflow.log_params(cfg)
        best_f1 = 0
        for epoch in range(epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            val_metrics = evaluate(model, test_loader, device, epoch)

            print(f"\nEpoch {epoch+1}:")
            print(f"  Train  - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, P: {train_metrics['precision']:.4f}, R: {train_metrics['recall']:.4f}")
            print(f"  Val    - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}\n")

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["f1"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"]
            }, step=epoch)

            early_stopping(val_metrics["loss"], model, model_path)
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), model_path)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        print(f"Training done. Best F1: {best_f1:.3f}")

if __name__ == "__main__":
    main()

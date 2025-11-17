import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from utils import EarlyStopping
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "cnn.yaml"
DATA_PATH = BASE_DIR / "Data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "cnn"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"
MLRUNS_DIR = BASE_DIR / "mlruns"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Load configs
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_cnn = yaml.safe_load(f)

# Load preprocessed data
X_train = np.load(DATA_PATH / "cnn_X_train.npy")
X_test = np.load(DATA_PATH / "cnn_X_test.npy")
y_train = np.load(DATA_PATH / "cnn_y_train.npy")
y_test = np.load(DATA_PATH / "cnn_y_test.npy")
vocab_info = np.load(DATA_PATH / "cnn_vocab_info.npy", allow_pickle=True).item()
vocab_size, pad_idx = vocab_info["vocab_size"], vocab_info["pad_idx"]

# Convert Ténor
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train,dtype=torch.long)
X_test_tensor = torch.tensor(X_test,dtype=torch.long)
y_test_tensor = torch.tensor(y_test,dtype=torch.long)

# Define Model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, dropout, pad_idx, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim), padding=(fs // 2, 0))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)         
        conv_outs = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.adaptive_max_pool1d(out, 1).squeeze(2) for out in conv_outs]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)


# Train function
def train_model(model, params, variant_name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    early_stopping = EarlyStopping(patience=params.get("patience", 5), verbose=True)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                              batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), 
                             batch_size=params["batch_size"], shuffle=False)

    experiment_name = f"vietnamese_sentiment_cnn_{variant_name}"
    mlflow.set_experiment(experiment_name)
    model_save_path = MODEL_DIR / f"{variant_name}_model.pt"

    with mlflow.start_run(run_name=variant_name):
        mlflow.log_params(params)

        for epoch in range(params["epochs"]):
            model.train()
            total_loss, preds_train, labels_train = 0, [], []

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                preds_train.extend(outputs.argmax(1).cpu().numpy())
                labels_train.extend(y_batch.cpu().numpy())

            avg_train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(labels_train, preds_train)
            train_f1 = f1_score(labels_train, preds_train, average="macro")

            # Validation
            model.eval()
            val_loss, preds, labels_all = 0, [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    preds.extend(outputs.argmax(1).cpu().numpy())
                    labels_all.extend(y_batch.cpu().numpy())

            avg_val_loss = val_loss / len(test_loader)
            val_acc = accuracy_score(labels_all, preds)
            val_precision = precision_score(labels_all, preds, average="macro", zero_division=0)
            val_recall = recall_score(labels_all, preds, average="macro", zero_division=0)
            val_f1 = f1_score(labels_all, preds, average="macro")

            print(f"[Epoch {epoch+1}/{params['epochs']}] "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_f1": train_f1,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            }, step=epoch)

            early_stopping(avg_val_loss, model, model_save_path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Load best model
        model.load_state_dict(torch.load(model_save_path))
        model.to("cpu")
        input_example = torch.tensor(X_train[:1])
        mlflow.pytorch.log_model(model, f"cnn_{variant_name}_model",
                                 input_example=input_example.numpy(),
                                 pip_requirements=str(REQUIREMENTS_PATH))
        model.to(device)
        print(f"Training completed for variant: {variant_name}\n")

# Train all variants
for variant_name, params in config_cnn["model_variants"].items():
    print(f"Starting training for {variant_name}...")
    cnn_model = CNNModel(
        vocab_size=vocab_size,
        embedding_dim=params["embedding_dim"],
        num_filters=params["num_filters"],
        filter_sizes=params["filter_sizes"],
        dropout=params["dropout"],
        pad_idx=pad_idx,
        num_classes=3  
    )
    train_model(cnn_model, params, variant_name)

print("All CNN model variants completed successfully!")
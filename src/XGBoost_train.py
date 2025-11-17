import numpy as np
import yaml
import mlflow
import mlflow.xgboost
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
from xgboost.callback import EarlyStopping

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "xgboost.yaml"
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "xgboost"
REQ_PATH = BASE_DIR / "requirements.txt"
MLRUNS_DIR = BASE_DIR / "mlruns"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


X_train = np.load(DATA_DIR / "xgboost_X_train.npy")
X_test = np.load(DATA_DIR / "xgboost_X_test.npy")
y_train = np.load(DATA_DIR / "xgboost_y_train.npy")
y_test = np.load(DATA_DIR / "xgboost_y_test.npy")



scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
scaler_path = MODEL_DIR / "xgboost_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")


mlflow.set_experiment("vietnamese_sentiment_xgboost")

# train all variants
for variant_name, params in config["model_variants"].items():
    print(f"\nTraining XGBoost variant: {variant_name}")
    with mlflow.start_run(run_name=variant_name):
        mlflow.log_params(params)

        params["num_class"] = 3
        params["objective"] = "multi:softmax"
        params["eval_metric"] = "mlogloss"

        # Build model
        use_gpu = torch.cuda.is_available()

        # Build model
        model = xgb.XGBClassifier(**params)

        # Train model
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1
        })

        # Save model and scaler
        model_path = MODEL_DIR / f"xgboost_{variant_name}.json"
        model.save_model(model_path)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=f"xgboost_{variant_name}_model",
            input_example=X_train[:1],
            pip_requirements=str(REQ_PATH)
        )

        print(f"Variant: {variant_name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

print("All XGBoost model variants completed successfully!")

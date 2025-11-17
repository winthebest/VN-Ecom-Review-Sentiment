# Vietnamese Sentiment Analysis Platform

End-to-end Vietnamese sentiment analysis stack with multiple model families (CNN, GRU, LSTM, XGBoost, PhoBERT). The project couples an automated training pipeline, MLflow tracking, and a production-ready Flask service for online inference.

## Highlights
- Vietnamese-specific preprocessing: teencode normalization, VnCoreNLP tokenization, stopword removal.
- Unified training pipeline (`pipeline.py`) that sequentially runs preprocessing and model training while logging metrics/artifacts to MLflow.
- Persisted datasets and checkpoints under `Data/processed/` and `models/` for quick reuse.
- PhoBERT inference API (`app/models/app.py`) exposing a `/predict` REST endpoint plus a lightweight HTML form.
- Dockerized runtime that ships the fine-tuned PhoBERT weights for portable deployment.

## Directory Layout
```
VN-Ecom-Review-Sentiment/
│── configs/            # YAML configuration files for different models
│   ├── cnn.yaml        # Config for CNN model
│   ├── gru.yaml        # Config for GRU model
│   ├── lstm.yaml       # Config for LSTM model
│   ├── xgboost.yaml    # Config for XGBoost model
│── Data/               # Raw and processed datasets
         ├── raw/       # Original dataset(s)
│        └── processed/ # Numpy/Torch tensors after preprocessing
│── models/             # Saved trained models
│── src/                # Source code 
│── pipeline.py         # Runs all training scripts sequentially
│── requirements.txt    # Required dependencies
│── Dockerfile          # Docker container setup
│── README.md           # Project documentation
│── app/
         ├── raw/              # FastAPI application
│           ├── app.py          # FastAPI server
│           ├── utils.py        # Helper functions
│           ├── Dockerfile      # Docker setup for API
│           ├── requirements.txt # Dependencies for API
│           ├── VnCoreNLP-master/ # VnCoreNLP module for NLP tasks
```

## Environment Setup
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r app/models/requirements.txt   # optional: isolated service env
   ```
3. (Optional) configure `JAVA_HOME` and download VnCoreNLP as outlined in `app/models/utils.py` if you rely on its word segmenter.
4. Pre-download PhoBERT into `cache/` to avoid hitting HuggingFace during training.

## Full Training Pipeline
```bash
python pipeline.py
```
- The script switches into `src/` and executes `preprocessing.py`, `CNN_train.py`, `GRU_train.py`, `LSTM_train.py`, `XGBoost_train.py`, `PhoBERT_train.py`.
- Inspect MLflow runs after training:
  ```bash
  mlflow ui --backend-store-uri file:///D:/Project/mlruns
  ```
  Visit `http://localhost:5000`.

### Train Models Individually
```bash
cd src
python preprocessing.py
python CNN_train.py
python GRU_train.py
python LSTM_train.py
python XGBoost_train.py
# PhoBERT_train.py can be enabled when you want to fine-tune again
```

## PhoBERT Inference Service
```bash
cd app/models
python app.py
```
- `POST /predict` accepts `{ "text": "..." }` and returns `NEG/NEU/POS` plus probabilities.
- `GET /` serves a minimal HTML form for quick manual checks.

## Docker Deployment
```bash
cd app/models
docker build -t vn-sentiment-phobert .
docker run -p 8000:8000 vn-sentiment-phobert
```
- Ensure `phobert_sentiment_classifier.pth` resides next to the Dockerfile before building.
- Push to Docker Hub (already done) via `docker tag` and `docker push` as needed.




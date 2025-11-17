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
D:/Project
├── app/models/            # Flask service + Docker context
├── configs/               # YAML configs per architecture
├── Data/
│   ├── raw/               # Original dataset(s)
│   └── processed/         # Numpy/Torch tensors after preprocessing
├── models/                # CNN/GRU/LSTM/XGBoost/PhoBERT checkpoints
├── mlruns/                # MLflow file-based tracking store
├── src/                   # Preprocessing + training scripts
└── pipeline.py            # Orchestrates full pipeline
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
- The script switches into `src/` and executes `preprocessing.py`, `CNN_train.py`, `GRU_train.py`, `LSTM_train.py`, `XGBoost_train.py`.
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

## Publishing to GitHub
1. Create a new GitHub repository (e.g., `vn-sentiment-analysis`) without auto-generated files.
2. On your local machine:
   ```bash
   cd D:/Project
   git init
   git add .
   git commit -m "Initial commit: Vietnamese sentiment analysis platform"
   git remote add origin https://github.com/<username>/vn-sentiment-analysis.git
   git push -u origin main
   ```
3. For an existing repo, update the remote via `git remote set-url origin ...` before pushing.

## Notes
- Large assets (`.pth`, `.npy`) are ignored via `.gitignore`. Provide alternative download links or regeneration instructions for collaborators.
- `mlruns/` grows quickly; consider external artifact stores or keep it local-only.
- For production, fix transformer/tokenizer versions and size instances (CPU/GPU) to guarantee reproducibility.


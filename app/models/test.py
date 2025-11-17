from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="phobert_sentiment_classifier.pth",
    path_in_repo="phobert_sentiment_classifier.pth",
    repo_id="tien2002/phobert-sentiment-v1"
)

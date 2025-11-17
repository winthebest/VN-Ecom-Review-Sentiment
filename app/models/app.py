from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
app = Flask(__name__)
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(
    repo_id="tien2002/phobert-sentiment-v1",
    filename="phobert_sentiment_classifier.pth"
)



tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)

model.load_state_dict(torch.load(model_path, map_location="cpu"))

LABELS = ["NEG", "NEU", "POS"]

# Define API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing text"}), 400

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label_idx = torch.argmax(probs, dim=1).item()
    
    result = {
        "text": text,
        "label": LABELS[label_idx],
        "probabilities": probs.tolist()
    }
    return jsonify(result)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text", "")
        if not text:
            return "nhập câu tiếng Việt!"

        # Tokenize và dự đoán
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            label_idx = torch.argmax(probs, dim=1).item()

        label = LABELS[label_idx]
        confidence = round(probs[0][label_idx].item() * 100, 2)

        return f"""
        <h2>Kết quả phân tích cảm xúc:</h2>
        <p><b>Câu nhập:</b> {text}</p>
        <p><b>Nhãn:</b> {label}</p>
        <p><b>Độ tin cậy:</b> {confidence}%</p>
        <br><a href='/'>← Phân tích câu khác</a>
        """

    return """
    <h2>Phân tích cảm xúc tiếng Việt (PhoBERT)</h2>
    <form method="POST" action="/">
        <input name="text" placeholder="Nhập câu tiếng Việt..." style="width:400px;padding:5px;">
        <input type="submit" value="Phân tích cảm xúc" style="padding:5px;">
    </form>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

import os
import re
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from nltk import word_tokenize
import py_vncorenlp
from py_vncorenlp import VnCoreNLP
from tqdm import tqdm
from pathlib import Path

os.environ["JAVA_HOME"] = r"C:\Users\ADMIN\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")
_vncorenlp_instance = None

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait before stopping after loss improvement.
            min_delta (float): Minimum change in loss to qualify as an improvement.
            verbose (bool): If True, prints message for each loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = float('inf')

    def __call__(self, val_loss, model, model_path):
        """
        Args:
            val_loss (float): Current validation loss
            model: PyTorch model to save if validation loss improves
            model_path (str): Path to save the model
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, model_path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.min_validation_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), model_path)
        self.min_validation_loss = val_loss

def load_vncorenlp():

    global _vncorenlp_instance

    if _vncorenlp_instance is not None:
        return _vncorenlp_instance 

    VNCORP_PATH = os.path.join(os.path.dirname(__file__), "VnCoreNLP")

    if not os.path.exists(VNCORP_PATH):
        print("Downloading VnCoreNLP model...")
        py_vncorenlp.download_model(save_dir=VNCORP_PATH)
    else:
        print(f"VnCoreNLP model folder already exists at {VNCORP_PATH}")

    _vncorenlp_instance = py_vncorenlp.VnCoreNLP(save_dir=VNCORP_PATH)
    return _vncorenlp_instance
'''
def load_phoBert():
    model = AutoModel.from_pretrained('vinai/phobert-base')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return model, tokenizer
'''
def load_phoBert():
    """
    Load PhoBERT model và tokenizer từ cache local để tránh tải lại từ internet.
    """
    cache_path = "D:/Project/cache" 
    print(f">>> [DEBUG] Loading PhoBERT from cache at {cache_path}")

    model = AutoModel.from_pretrained("vinai/phobert-base", cache_dir=cache_path)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False, cache_dir=cache_path)
    print(">>> [DEBUG] PhoBERT loaded successfully from cache.")
    return model, tokenizer
    
def load_vn_teencode_dict():
    # Lấy đường dẫn tương đối tính từ file utils.py
    base_dir = Path(__file__).resolve().parent.parent  # /app/src → /app
    dict_path = base_dir / "Data" / "vi-nsw-dict.json"

    if not dict_path.exists():
        raise FileNotFoundError(f"Dictionary not found at {dict_path}")

    with open(dict_path, "r", encoding="utf-8") as f:
        teencode_dict = json.load(f)
    return teencode_dict
    

def standardize_data(text: str):
    """Lowercase, remove punctuation, numbers, emojis."""
    if not isinstance(text, str):
        text = str(text)
    pattern = r"[^a-zA-ZÀ-Ỹà-ỹ\s]"  
    text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()  #
    return text.lower()


def remove_repetitive_characters(text: str):
    # Define regex pattern to match repetitive characters
    return re.sub(r"(.)\1{2,}", r"\1", text) # Match one character followed by one or more occurrences of the same character


def correct_spelling_teencode(text: str, teencode_dict, vn_segmenter):
    """
    Chuẩn hoá teencode và từ sai chính tả, 
    đồng thời tách từ bằng VnCoreNLP để tương thích với PhoBERT.
    """
    text = text.lower().strip()
    segmented_words = vn_segmenter.word_segment(text)  # -> list of token lists
    flattened_words = [word for sent in segmented_words for word in sent]

    corrected_words = [
        teencode_dict[word][0] if word in teencode_dict else word
        for word in flattened_words
    ]
    return " ".join(corrected_words)


def load_stopwords(stopword_path="D:/Project/Data/stopwords-vi.txt"):
    try:
        with open(stopword_path, "r", encoding="utf-8") as f:
            stopwords = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(stopwords)} Vietnamese stopwords.")
        return stopwords
    except FileNotFoundError:
        print(f"[Warning] Stopword file not found at {stopword_path}. Continuing without stopword removal.")
        return set()


def remove_stopwords(text: str, stopwords: set):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stopwords]
    return " ".join(filtered_tokens)



def extract_phobert_features(df, text_column, label_column=None, max_len=64, batch_size=256, vn_segmenter=None):
    """
    Process text data through preprocessing, PhoBERT tokenization, and feature extraction
    with memory-efficient batch processing
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing text data
    text_column : str
        Name of the column containing text to process
    label_column : str, optional
        Name of the column containing labels
    max_len : int, default=100
        Maximum sequence length for tokenization
    batch_size : int, default=32
        Batch size for GPU processing. Reduced to prevent VRAM overflow.
        
    Returns:
    --------
    tuple
        (features array, labels array if label_column provided)
    """
    print("Entering extract_phobert_features()")
    if vn_segmenter is None:
        vn_segmenter = load_vncorenlp()
    # Preprocessing
    df = df.copy()

    # Get data
    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip() != ""]
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].values if label_column else None

    # Load models once
    phobert, tokenizer = load_phoBert()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    phobert = phobert.to(device)
    phobert.eval()

    all_features = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting PhoBERT features"):
        print(f"Processing batch {i // batch_size + 1}/{len(texts)//batch_size + 1}")
        batch_texts = texts[i:i+batch_size]

        # Word segmentation
        segmented = []
        for t in batch_texts:
            t = t.strip()
            if not t:
                segmented.append("")  # skip if it empty
            else:
                try:
                    #seg = vn_segmenter.word_segment(t)
                    #segmented.append(" ".join(seg))
                    seg = vn_segmenter.word_segment(t)
                    segmented.append(" ".join([word for sent in seg for word in sent]))

                except Exception as e:
                    print(f"Skipping problematic text: {t[:50]}... ({e})")
                    segmented.append("")


        # Tokenize
        encodings = tokenizer(
            segmented,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        # Extract [CLS] token embedding
        with torch.no_grad():
            outputs = phobert(**encodings)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_features.append(cls_embeddings)
        torch.cuda.empty_cache()

    # Combine all batches
    features = np.concatenate(all_features, axis=0)
    print(f"PhoBERT feature extraction done. Shape = {features.shape}")

    return (features, labels) if label_column else features
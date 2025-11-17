import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from utils import (
    remove_repetitive_characters, 
    correct_spelling_teencode, 
    standardize_data,
    extract_phobert_features,
    load_vn_teencode_dict,
    load_vncorenlp,
    load_stopwords,
    remove_stopwords
)

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)


def preprocess_data(input_path, output_dir, model_type="xgboost"):
    df = pd.read_csv(input_path)

    df = df[["comment", "label"]].dropna()

    # Convert sentiment to binary labels
    label_mapping = {"NEG": 0, "NEU": 1, "POS": 2}
    df["label"] = df["label"].map(label_mapping)

    # Load dictionary
    teencode_dict = load_vn_teencode_dict()
    vn_segmenter = load_vncorenlp()
    stopwords = load_stopwords("D:/Project/Dataset/stopwords-vi.txt") 
    df["processed_comment"] = (
        df["comment"]
        .astype(str)
        .apply(standardize_data)
        .apply(lambda x: remove_repetitive_characters(x))
        .apply(lambda x: correct_spelling_teencode(x, teencode_dict, vn_segmenter))
        .apply(lambda x: remove_stopwords(x, stopwords))
    )

    # Remove empty rows after preprocessing
    df = df[df["processed_comment"].str.strip() != ""]
    print(f"Cleaned dataset: {len(df)} samples")

    os.makedirs(output_dir, exist_ok=True)

    # Specific preprocessing for XGBoost
    if model_type == "xgboost":
         # Use PhoBERT features
        X, y = extract_phobert_features(df, "processed_comment", "label", vn_segmenter=vn_segmenter)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )


    # Preprocessing for Deep Learning models    
    elif model_type in ["lstm", "gru", "cnn"]:
        import nltk
        nltk.download("punkt", quiet=True)
        tokenizer = get_tokenizer("basic_english")
        # Build vocab
        vocab = build_vocab_from_iterator(
            yield_tokens(df["processed_comment"], tokenizer),
            min_freq=2,
            specials=["<unk>", "<pad>"]
        )
        vocab.set_default_index(vocab["<unk>"])

        # Save vocab
        torch.save(vocab, os.path.join(output_dir, f"{model_type}_vocab.pth"))

        # Tokenize and convert text into vocabulary indices
        max_len = 64

        def text_pipeline(text):
            tokens = tokenizer(text)
            indices = [vocab[token] for token in tokens]
            if len(indices) < max_len:
                indices += [vocab["<pad>"]] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
            return indices

        X = np.array([text_pipeline(text) for text in df["processed_comment"]])
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Save infomative vocab
        vocab_info = {
            "vocab_size": len(vocab),
            "max_len": max_len,
            "pad_idx": vocab["<pad>"]
        }
        np.save(os.path.join(output_dir, f"{model_type}_vocab_info.npy"), vocab_info)

    np.save(os.path.join(output_dir, f"{model_type}_X_train.npy"), X_train)
    np.save(os.path.join(output_dir, f"{model_type}_X_test.npy"), X_test)
    np.save(os.path.join(output_dir, f"{model_type}_y_train.npy"), y_train)
    np.save(os.path.join(output_dir, f"{model_type}_y_test.npy"), y_test)

    print(f"Saved preprocessed data for {model_type}")
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent  
    base_data_path = BASE_DIR / "Data" / "raw" / "labeled_data.csv"
    output_data_path = BASE_DIR / "Data" / "processed"

    # Preprocessing for different model types
    preprocess_data(base_data_path, output_data_path, model_type="xgboost")
    preprocess_data(base_data_path, output_data_path, model_type="lstm")
    preprocess_data(base_data_path, output_data_path, model_type="gru")
    preprocess_data(base_data_path, output_data_path, model_type="cnn")
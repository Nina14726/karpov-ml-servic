from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertModel

from load_data import load_all_data, validate_loaded_data


DATABASE_URL = os.getenv("DATABASE_URL")
POST_FEATURES_TABLE = "nina14726_post_features_dl"
MODEL_PATH = Path(__file__).with_name("model.pkl")

CAT_FEATURES = ["gender", "country", "city", "exp_group", "os", "source", "topic"]
EMB_COLS = [f"emb_{i}" for i in range(15)]
FEATURES = (
    CAT_FEATURES
    + ["age", "text_length", "word_count", "unique_word_count", "hour", "dayofweek", "month"]
    + EMB_COLS
)


def make_user_features(users: pd.DataFrame) -> pd.DataFrame:
    result = users.copy()
    for column in ["gender", "country", "city", "exp_group", "os", "source"]:
        result[column] = result[column].astype(str)
    result["age"] = pd.to_numeric(result["age"], downcast="integer")
    return result


@torch.inference_mode()
def get_distilbert_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    model = DistilBertModel.from_pretrained("distilbert-base-cased").to(device)
    model.eval()

    batches = []
    for start in tqdm(range(0, len(texts), batch_size), desc="DistilBERT"):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        embeddings = model(**encoded)["last_hidden_state"][:, 0, :]
        batches.append(embeddings.cpu().numpy().astype("float32"))

    return np.vstack(batches)


def make_post_features(posts: pd.DataFrame) -> pd.DataFrame:
    result = posts.copy()
    text = result["text"].fillna("").astype(str)

    result["text_length"] = text.str.len().astype("int32")
    result["word_count"] = text.str.split().str.len().astype("int32")
    result["unique_word_count"] = text.str.lower().str.split().map(lambda words: len(set(words))).astype("int32")

    bert_embeddings = get_distilbert_embeddings(text.tolist())
    reduced = PCA(n_components=15, random_state=42).fit_transform(bert_embeddings).astype("float32")

    for i, column in enumerate(EMB_COLS):
        result[column] = reduced[:, i]

    result["topic"] = result["topic"].astype(str)
    return result[["post_id", "topic", "text_length", "word_count", "unique_word_count"] + EMB_COLS]


def build_training_data(users: pd.DataFrame, posts: pd.DataFrame, feed: pd.DataFrame):
    user_features = make_user_features(users)
    post_features = make_post_features(posts)

    interactions = feed.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions["user_id"] = pd.to_numeric(interactions["user_id"], downcast="integer")
    interactions["post_id"] = pd.to_numeric(interactions["post_id"], downcast="integer")
    interactions["target"] = pd.to_numeric(interactions["target"], downcast="integer")
    interactions["hour"] = interactions["timestamp"].dt.hour.astype("int8")
    interactions["dayofweek"] = interactions["timestamp"].dt.dayofweek.astype("int8")
    interactions["month"] = interactions["timestamp"].dt.month.astype("int8")

    dataset = interactions.merge(user_features, on="user_id", how="inner")
    dataset = dataset.merge(post_features, on="post_id", how="inner")
    dataset = dataset.drop(columns=["action", "user_id", "post_id"])

    if dataset.empty:
        raise ValueError("После объединения таблиц обучающая выборка пуста")

    return dataset, post_features


def split_by_time(dataset: pd.DataFrame, test_fraction: float = 0.2):
    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()

    if train.empty or test.empty:
        raise ValueError("Не удалось сформировать train/test")

    return train, test


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame):
    X_train = train[FEATURES]
    y_train = train["target"].astype("int8")
    X_test = test[FEATURES]
    y_test = test["target"].astype("int8")

    model = CatBoostClassifier(
        iterations=700,
        depth=8,
        learning_rate=0.07,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=CAT_FEATURES,
        eval_set=(X_test, y_test),
        use_best_model=True,
        early_stopping_rounds=70,
    )

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, train_auc, test_auc


def save_model(model: CatBoostClassifier) -> None:
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)


def save_post_features(post_features: pd.DataFrame) -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(DATABASE_URL)
    try:
        post_features.to_sql(
            POST_FEATURES_TABLE,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=500,
        )
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-limit", type=int, default=5_000_000)
    parser.add_argument("--skip-save-features", action="store_true")
    args = parser.parse_args()

    users, posts, feed = load_all_data(args.feed_limit)
    validate_loaded_data(users, posts, feed)

    dataset, post_features = build_training_data(users, posts, feed)
    train, test = split_by_time(dataset)
    model, train_auc, test_auc = train_and_evaluate(train, test)

    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")

    save_model(model)
    print(f"Модель сохранена: {MODEL_PATH}")

    if not args.skip_save_features:
        save_post_features(post_features)
        print(f"DL-признаки постов сохранены: {POST_FEATURES_TABLE}")


if __name__ == "__main__":
    main()

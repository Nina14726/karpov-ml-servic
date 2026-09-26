from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine

from load_data import load_all_data, validate_loaded_data


DATABASE_URL = os.getenv("DATABASE_URL")
USER_FEATURES_TABLE = "nina14726_user_features"
POST_FEATURES_TABLE = "nina14726_post_features"
MODEL_PATH = Path(__file__).with_name("model.pkl")


def make_user_features(users: pd.DataFrame) -> pd.DataFrame:
    result = users.copy()
    result["user_id"] = result["user_id"].astype(str)
    result["exp_group"] = result["exp_group"].astype(str)
    return result


def len_of_unique_words(words: list[str]) -> int:
    return len(set(words))


def make_post_features(posts: pd.DataFrame) -> pd.DataFrame:
    result = posts.rename(columns={"id": "post_id"}).copy()
    text = result["text"].fillna("").astype(str)
    result["text_length"] = text.str.len()
    result["word_count"] = text.str.split().str.len()
    result["unique_word_count"] = text.str.lower().str.split().map(len_of_unique_words)
    result["post_id"] = result["post_id"].astype(str)
    return result.drop(columns="text")


def build_training_data(users, posts, feed):
    user_features = make_user_features(users)
    post_features = make_post_features(posts)

    interactions = feed.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["post_id"] = interactions["post_id"].astype(str)
    interactions["hour"] = interactions["timestamp"].dt.hour.astype(str)
    interactions["day_of_week"] = interactions["timestamp"].dt.dayofweek.astype(str)
    interactions["month"] = interactions["timestamp"].dt.month.astype(str)

    dataset = interactions.merge(user_features, on="user_id", how="inner")
    dataset = dataset.merge(post_features, on="post_id", how="inner")
    dataset = dataset.drop(columns="action")

    if dataset.empty:
        raise ValueError("После объединения таблиц обучающая выборка пуста")

    return dataset, user_features, post_features


def split_by_time(dataset: pd.DataFrame, test_fraction: float = 0.2):
    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame):
    feature_columns = [c for c in train.columns if c not in {"target", "timestamp"}]
    categorical_columns = [
        "user_id", "post_id", "gender", "country", "city", "exp_group",
        "os", "source", "topic", "hour", "day_of_week", "month"
    ]

    X_train = train[feature_columns]
    y_train = train["target"].astype(int)
    X_test = test[feature_columns]
    y_test = test["target"].astype(int)

    model = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.08,
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
        cat_features=categorical_columns,
        eval_set=(X_test, y_test),
        use_best_model=True,
        early_stopping_rounds=50,
    )

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, train_auc, test_auc


def save_model(model: CatBoostClassifier, path: Path = MODEL_PATH) -> None:
    with path.open("wb") as file:
        pickle.dump(model, file)


def save_features(user_features: pd.DataFrame, post_features: pd.DataFrame) -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    engine = create_engine(DATABASE_URL)
    try:
        user_features.to_sql(USER_FEATURES_TABLE, engine, if_exists="replace", index=False, method="multi", chunksize=1000)
        post_features.to_sql(POST_FEATURES_TABLE, engine, if_exists="replace", index=False, method="multi", chunksize=1000)
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-limit", type=int, default=5_000_000)
    parser.add_argument("--skip-save-features", action="store_true")
    args = parser.parse_args()

    users, posts, feed = load_all_data(feed_limit=args.feed_limit)
    validate_loaded_data(users, posts, feed)
    dataset, user_features, post_features = build_training_data(users, posts, feed)
    train, test = split_by_time(dataset)
    model, train_auc, test_auc = train_and_evaluate(train, test)

    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    save_model(model)

    if not args.skip_save_features:
        save_features(user_features, post_features)


if __name__ == "__main__":
    main()

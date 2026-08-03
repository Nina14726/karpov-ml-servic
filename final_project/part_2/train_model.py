from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine

from load_data import load_all_data, validate_loaded_data


DATABASE_URL = (
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)
USER_FEATURES_TABLE = "nina14726_user_features"
POST_FEATURES_TABLE = "nina14726_post_features"
MODEL_PATH = Path(__file__).with_name("model.pkl")


def make_user_features(users: pd.DataFrame) -> pd.DataFrame:
    """Готовит признаки профиля пользователя."""
    result = users.copy()
    result["user_id"] = result["user_id"].astype(str)
    result["exp_group"] = result["exp_group"].astype(str)
    return result


def make_post_features(posts: pd.DataFrame) -> pd.DataFrame:
    """Готовит признаки темы и текста публикации."""
    result = posts.rename(columns={"id": "post_id"}).copy()
    text = result["text"].fillna("").astype(str)

    result["text_length"] = text.str.len()
    result["word_count"] = text.str.split().str.len()
    result["unique_word_count"] = text.str.lower().str.split().map(len_of_unique_words)
    result["post_id"] = result["post_id"].astype(str)

    return result.drop(columns="text")


def len_of_unique_words(words: list[str]) -> int:
    return len(set(words))


def build_training_data(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    feed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Объединяет взаимодействия с признаками пользователя, поста и времени."""
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


def split_by_time(
    dataset: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет данные по времени: последние события используются для проверки."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction должен находиться между 0 и 1")

    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()

    if train.empty or test.empty:
        raise ValueError("Не удалось сформировать train/test выборки")
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise ValueError("В train и test должны присутствовать оба класса target")

    return train, test


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[CatBoostClassifier, float, float]:
    """Обучает CatBoost и рассчитывает ROC-AUC на train и test."""
    feature_columns = [
        column
        for column in train.columns
        if column not in {"target", "timestamp"}
    ]
    categorical_columns = [
        "user_id",
        "post_id",
        "gender",
        "country",
        "city",
        "exp_group",
        "os",
        "source",
        "topic",
        "hour",
        "day_of_week",
        "month",
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
    """Сохраняет обученную модель в pickle и проверяет обратную загрузку."""
    with path.open("wb") as file:
        pickle.dump(model, file)

    with path.open("rb") as file:
        loaded_model = pickle.load(file)

    if not hasattr(loaded_model, "predict"):
        raise TypeError("В сохранённой модели отсутствует метод predict")
    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError("В сохранённой модели отсутствует метод predict_proba")


def save_features(
    user_features: pd.DataFrame,
    post_features: pd.DataFrame,
) -> None:
    """Сохраняет только признаки пользователей и постов в PostgreSQL."""
    engine = create_engine(DATABASE_URL)
    try:
        user_features.to_sql(
            USER_FEATURES_TABLE,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
        post_features.to_sql(
            POST_FEATURES_TABLE,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подготовка признаков, обучение и сохранение модели"
    )
    parser.add_argument(
        "--feed-limit",
        type=int,
        default=5_000_000,
        help="Количество просмотров из public.feed_data (не более 10 000 000)",
    )
    parser.add_argument(
        "--skip-save-features",
        action="store_true",
        help="Не сохранять признаки в PostgreSQL",
    )
    args = parser.parse_args()

    users, posts, feed = load_all_data(feed_limit=args.feed_limit)
    validate_loaded_data(users, posts, feed)

    dataset, user_features, post_features = build_training_data(users, posts, feed)
    train, test = split_by_time(dataset)
    model, train_auc, test_auc = train_and_evaluate(train, test)

    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")

    if test_auc <= 0.5:
        raise ValueError("ROC-AUC на тестовой выборке должен быть выше 0.5")

    save_model(model)
    print(f"Модель сохранена: {MODEL_PATH}")

    if not args.skip_save_features:
        save_features(user_features, post_features)
        print(f"Признаки пользователей сохранены в {USER_FEATURES_TABLE}")
        print(f"Признаки постов сохранены в {POST_FEATURES_TABLE}")


if __name__ == "__main__":
    main()

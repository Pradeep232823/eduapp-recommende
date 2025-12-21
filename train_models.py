import pandas as pd
import numpy as np
import pickle
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model


# -------------------------------
# CONTENT-BASED MODEL
# -------------------------------
def train_content_based(df_apps):
    print("Training Content-Based Model...")

    df_apps["description"] = df_apps["description"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_apps["description"])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    with open("cosine_sim.pkl", "wb") as f:
        pickle.dump(cosine_sim, f)

    indices = pd.Series(df_apps.index, index=df_apps["app_name"]).drop_duplicates()
    with open("indices.pkl", "wb") as f:
        pickle.dump(indices, f)

    print("Content-Based Model trained and saved.")


# -------------------------------
# COLLABORATIVE FILTERING MODEL
# -------------------------------
def train_collaborative_filtering(df_ratings):
    print("Training Collaborative Filtering Model...")

    user_ids = df_ratings["user_id"].unique()
    app_ids = df_ratings["app_id"].unique()

    user_encoder = {u: i for i, u in enumerate(user_ids)}
    app_encoder = {a: i for i, a in enumerate(app_ids)}

    df_ratings["user"] = df_ratings["user_id"].map(user_encoder)
    df_ratings["app"] = df_ratings["app_id"].map(app_encoder)

    num_users = len(user_encoder)
    num_apps = len(app_encoder)

    # Train–test split (BASE PAPER REQUIREMENT)
    train_df, test_df = train_test_split(
        df_ratings, test_size=0.2, random_state=42
    )

    X_train = train_df[["user", "app"]].values
    y_train = train_df["rating"].values

    X_test = test_df[["user", "app"]].values
    y_test = test_df["rating"].values

    # Model Architecture (Neural CF)
    user_input = Input(shape=(1,))
    app_input = Input(shape=(1,))

    user_embedding = Embedding(num_users, 50)(user_input)
    app_embedding = Embedding(num_apps, 50)(app_input)

    user_vec = Flatten()(user_embedding)
    app_vec = Flatten()(app_embedding)

    concat = Concatenate()([user_vec, app_vec])
    dense = Dense(128, activation="relu")(concat)
    dense = Dense(64, activation="relu")(dense)
    output = Dense(1)(dense)

    model = Model(inputs=[user_input, app_input], outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=64,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    # Evaluation (BASE PAPER REQUIREMENT)
    predictions = model.predict([X_test[:, 0], X_test[:, 1]])
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE on test set: {rmse:.4f}")

    model.save("cf_model.h5")

    with open("user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open("app_encoder.pkl", "wb") as f:
        pickle.dump(app_encoder, f)

    print("Collaborative Filtering Model trained, evaluated, and saved.")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    df_apps = pd.read_csv("apps.csv")
    df_ratings = pd.read_csv("ratings.csv")

    train_content_based(df_apps)
    train_collaborative_filtering(df_ratings)

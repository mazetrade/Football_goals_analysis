import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, build_features

RAW_PATH = "data/raw/shots.pkl"
MODEL_PATH = "models/xg_model.joblib"


def build_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def train_model(df):
    model = build_model()
    model.fit(df[FEATURES], df["is_goal"])
    return model


if __name__ == "__main__":
    raw = pd.read_pickle(RAW_PATH)
    df = build_features(raw)
    model = train_model(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle entraîné sur {len(df)} tirs et sauvegardé dans {MODEL_PATH}")
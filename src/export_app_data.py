import pandas as pd
from sklearn.model_selection import cross_val_predict

from features import FEATURES, build_features
from train import build_model

RAW_PATH = "data/raw/shots.pkl"
APP_DATA_PATH = "app/shots_xg.csv"

COLUMNS = [
    "player", "team", "competition", "minute",
    "x", "y", "distance", "angle", "is_header",
    "is_goal", "xg", "shot_statsbomb_xg",
]


if __name__ == "__main__":
    df = build_features(pd.read_pickle(RAW_PATH))
    df["xg"] = cross_val_predict(
        build_model(), df[FEATURES], df["is_goal"], cv=5, method="predict_proba"
    )[:, 1]
    df[COLUMNS].to_csv(APP_DATA_PATH, index=False)
    print(f"{len(df)} tirs exportés dans {APP_DATA_PATH}")
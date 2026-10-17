# Imports

import warnings
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

COMPETITIONS = [
    ("FIFA World Cup", "2018"),
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2020"),
    ("UEFA Euro", "2024"),
]

SHOT_COLUMNS = [
    "id", "match_id", "minute", "player", "team",
    "location", "shot_body_part", "shot_type", "shot_technique",
    "play_pattern", "under_pressure", "shot_first_time",
    "shot_one_on_one", "shot_open_goal", "shot_freeze_frame",
    "shot_outcome", "shot_statsbomb_xg",
]

# Find competitions ID

def get_ids(competition_name, season_name):
    comps = sb.competitions()
    row = comps[
        (comps["competition_name"] == competition_name)
        & (comps["season_name"] == season_name)
    ]
    if row.empty:
        raise ValueError(f"Compétition introuvable : {competition_name} {season_name}")
    return int(row["competition_id"].iloc[0]), int(row["season_id"].iloc[0])

# Load competition shots 

def load_shots(competition_name, season_name):
    comp_id, season_id = get_ids(competition_name, season_name)
    matches = sb.matches(competition_id=comp_id, season_id=season_id)

    all_shots = []
    for i, match_id in enumerate(matches["match_id"], start=1):
        events = sb.events(match_id=match_id)
        shots = events[events["type"] == "Shot"].reindex(columns=SHOT_COLUMNS)
        shots["match_id"] = match_id
        all_shots.append(shots)
        print(f"  {competition_name} {season_name} : match {i}/{len(matches)}", end="\r")

    print()
    df = pd.concat(all_shots, ignore_index=True)
    df["competition"] = f"{competition_name} {season_name}"
    return df

# Save 

def build_dataset(save_path="data/raw/shots.pkl"):
    frames = [load_shots(name, season) for name, season in COMPETITIONS]
    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_pickle(save_path)
    print(f"{len(dataset)} tirs sauvegardés dans {save_path}")
    return dataset


if __name__ == "__main__":
    build_dataset()
    

# Targeted variables

import numpy as np
import pandas as pd

GOAL_X = 120
GOAL_Y = 40
GOAL_WIDTH = 8
POST_LEFT = (GOAL_X, GOAL_Y - GOAL_WIDTH / 2)
POST_RIGHT = (GOAL_X, GOAL_Y + GOAL_WIDTH / 2)

FEATURES = [
    "distance", "angle",
    "is_header", "under_pressure", "shot_first_time",
    "shot_one_on_one", "shot_open_goal",
    "n_def_in_cone", "n_def_close", "gk_dist_to_goal",
]


def add_target(df):
    df = df.copy()
    df["is_goal"] = (df["shot_outcome"] == "Goal").astype(int)
    return df

# Angles and distance

def add_distance_angle(df):
    df = df.copy()
    df["x"] = df["location"].apply(lambda loc: loc[0])
    df["y"] = df["location"].apply(lambda loc: loc[1])

    dx = GOAL_X - df["x"]
    dy = (GOAL_Y - df["y"]).abs()

    df["distance"] = np.sqrt(dx**2 + dy**2)

    angle_rad = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2)
    df["angle"] = np.degrees(angle_rad)
    return df

# Remove penalties 

def remove_penalties(df):
    return df[df["shot_type"] != "Penalty"].copy()

# Find boolean columns 

BOOL_COLUMNS = ["under_pressure", "shot_first_time", "shot_one_on_one", "shot_open_goal"]


def clean_boolean_flags(df):
    df = df.copy()
    for col in BOOL_COLUMNS:
        df[col] = df[col].eq(True).astype(int)
    return df

# Body parts 

def add_body_part(df):
    df = df.copy()
    df["is_header"] = df["shot_body_part"].eq("Head").astype(int)
    return df

# Bring all features together 

def build_features(df):
    df = add_target(df)
    df = remove_penalties(df)
    df = add_distance_angle(df)
    df = clean_boolean_flags(df)
    df = add_body_part(df)
    df = add_freeze_frame_features(df)
    return df

# Point in triangle 

def point_in_triangle(p, a, b, c):
    def side(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = side(p, a, b)
    d2 = side(p, b, c)
    d3 = side(p, c, a)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)

# Freeze frame stats 

def freeze_frame_stats(frame, shooter):
    stats = {"n_def_in_cone": 0, "n_def_close": 0, "gk_dist_to_goal": np.nan}

    if not isinstance(frame, list):
        return stats

    for player in frame:
        if player["teammate"]:
            continue
        x, y = player["location"]
        if player["position"]["name"] == "Goalkeeper":
            stats["gk_dist_to_goal"] = np.hypot(GOAL_X - x, GOAL_Y - y)
            continue
        if point_in_triangle((x, y), shooter, POST_LEFT, POST_RIGHT):
            stats["n_def_in_cone"] += 1
        if np.hypot(x - shooter[0], y - shooter[1]) < 3:
            stats["n_def_close"] += 1

    return stats

def add_freeze_frame_features(df):
    df = df.copy()
    stats = df.apply(
        lambda row: freeze_frame_stats(row["shot_freeze_frame"], row["location"]),
        axis=1,
        result_type="expand",
    )
    return df.join(stats)
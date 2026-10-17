import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import streamlit as st
from mplsoccer import VerticalPitch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

APP_DIR = Path(__file__).parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURES, POST_LEFT, POST_RIGHT, add_distance_angle  # noqa: E402

DATA_PATH = APP_DIR / "shots_xg.csv"
MODEL_PATH = ROOT / "models" / "xg_model.joblib"

# ---------------------------------------------------------------- palette
TURF = "#0F3D26"
TURF_DARK = "#0A2E1C"
STRIPE = "#145232"
CHALK = "#F2F5EC"
SAGE = "#A9C7B2"
GOLD = "#F2C14E"
RED = "#E8735A"

XG_CMAP = LinearSegmentedColormap.from_list(
    "xg", ["#F2C14E00", "#F2C14E80", "#E8735AD0", "#B0302AF0"]
)

FEATURE_LABELS = {
    "distance": "Distance to goal",
    "angle": "Shooting angle",
    "is_header": "Header",
    "under_pressure": "Under pressure",
    "shot_first_time": "First-time shot",
    "shot_one_on_one": "One-on-one",
    "shot_open_goal": "Open goal",
    "n_def_in_cone": "Defenders in the shot cone",
    "n_def_close": "Defenders within 3 yards",
    "gk_dist_to_goal": "Keeper distance from goal",
}

st.set_page_config(page_title="xG Lab", page_icon="⚽", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, p, label {{ font-family: 'Barlow', sans-serif; }}
h1, h2, h3, [data-testid="stMetricValue"] {{
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.01em;
}}
.hero h1 {{ font-size: 3.4rem; line-height: 1.05; margin: 0 0 .4rem 0; color: {CHALK}; }}
.hero p {{ font-size: 1.15rem; color: {SAGE}; max-width: 60ch; margin: 0; }}
.hero {{ border-left: 6px solid {GOLD}; padding: .4rem 0 .4rem 1.2rem; margin-bottom: 1.6rem; }}
[data-testid="stMetric"] {{
    border: 1px solid rgba(242, 245, 236, .14);
    border-radius: 10px;
    padding: 12px 16px;
    background: {TURF_DARK};
}}
[data-testid="stMetricValue"] {{ color: {GOLD}; font-size: 2.3rem; }}
.note {{
    background: {TURF_DARK};
    border-left: 4px solid {GOLD};
    padding: .8rem 1rem;
    color: {CHALK};
    margin-top: .6rem;
}}
.verdict {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.6rem; color: {CHALK}; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
@st.cache_data
def load_shots():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def xg_surface(_model, flags):
    xs, ys = np.meshgrid(np.arange(60.5, 120, 1.0), np.arange(0.5, 80, 1.0))
    grid = pd.DataFrame({"location": list(zip(xs.ravel(), ys.ravel()))})
    for name, value in flags.items():
        grid[name] = value
    grid = add_distance_angle(grid)
    grid["xg"] = _model.predict_proba(grid[FEATURES])[:, 1]
    return grid[["x", "y", "xg"]]


def predict_xg(model, x, y, flags):
    shot = add_distance_angle(pd.DataFrame([{"location": (x, y), **flags}]))
    return float(model.predict_proba(shot[FEATURES])[0, 1])


def describe_chance(xg):
    if xg < 0.05:
        return "Speculative effort"
    if xg < 0.15:
        return "Half-chance"
    if xg < 0.35:
        return "Good chance"
    return "Big chance"


# ---------------------------------------------------------------- drawing
def make_pitch(figsize=(8, 7)):
    pitch = VerticalPitch(
        pitch_type="statsbomb", half=True, pitch_color=TURF, line_color=CHALK,
        stripe=True, stripe_color=STRIPE, linewidth=1.4, goal_type="box",
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(TURF)
    return pitch, fig, ax


def style_legend(ax):
    legend = ax.legend(loc="lower left", facecolor=TURF_DARK, edgecolor=TURF_DARK,
                       labelcolor=CHALK, fontsize=11)
    for handle in legend.legend_handles:
        handle.set_sizes([140])


def draw_shot_map(shots_df, title):
    pitch, fig, ax = make_pitch()
    goals = shots_df[shots_df["is_goal"] == 1]
    misses = shots_df[shots_df["is_goal"] == 0]
    pitch.scatter(misses["x"], misses["y"], s=misses["xg"] * 900 + 30, ax=ax,
                  facecolors="none", edgecolors=CHALK, linewidths=1.1, alpha=0.65,
                  label="No goal")
    pitch.scatter(goals["x"], goals["y"], s=goals["xg"] * 900 + 30, ax=ax,
                  color=GOLD, edgecolors=TURF_DARK, linewidths=1, zorder=3, label="Goal")
    style_legend(ax)
    ax.set_title(title, color=CHALK, fontsize=16, pad=10)
    return fig


def draw_simulator_pitch(surface, x, y):
    pitch, fig, ax = make_pitch()
    stats = pitch.bin_statistic(surface["x"], surface["y"], values=surface["xg"],
                                statistic="mean", bins=(120, 80))
    pitch.heatmap(stats, ax=ax, cmap=XG_CMAP, vmin=0, vmax=0.5, zorder=1)
    pitch.polygon([[(x, y), POST_LEFT, POST_RIGHT]], ax=ax, color=CHALK, alpha=0.3, zorder=3)
    pitch.scatter(x, y, s=380, ax=ax, color=GOLD, edgecolors=TURF_DARK, linewidths=2, zorder=4)
    return fig


def style_axes(fig, ax):
    fig.set_facecolor(TURF)
    ax.set_facecolor(TURF)
    ax.tick_params(colors=CHALK)
    ax.xaxis.label.set_color(CHALK)
    ax.yaxis.label.set_color(CHALK)
    ax.title.set_color(CHALK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(color=STRIPE, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_bars(values, color, title):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(values.index, values.values, color=color)
    ax.invert_yaxis()
    ax.axvline(0, color=SAGE, linewidth=1)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(axis="y", visible=False)
    style_axes(fig, ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- page
shots = load_shots()
model = load_model()

st.markdown(
    """
<div class="hero">
  <h1>How likely was that to go in?</h1>
  <p>An expected goals model built from scratch on StatsBomb open data:
  every non-penalty shot from the 2018 and 2022 World Cups and Euro 2020 and 2024.</p>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Shots analysed", f"{len(shots):,}")
m2.metric("Non-penalty goals", int(shots["is_goal"].sum()))
m3.metric("Model xG", f"{shots['xg'].sum():.1f}")
m4.metric("Players", shots["player"].nunique())

tab_sim, tab_map, tab_board, tab_model = st.tabs(
    ["🎯 Shot simulator", "🗺️ Shot maps", "🏆 Finishing leaderboard", "🧠 How the model works"]
)

# ---------------------------------------------------------------- simulator
with tab_sim:
    st.subheader("Set up a shot and see its xG")
    controls, view = st.columns([1, 1.5], gap="large")

    with controls:
        depth = st.slider("Distance from the goal line (yards)", 1, 40, 12)
        lateral = st.slider("Position across the pitch (yards from centre, left − / right +)",
                            -30, 30, 0)
        body = st.radio("Body part", ["Foot", "Head"], horizontal=True)
        c1, c2 = st.columns(2)
        first_time = c1.checkbox("First-time shot")
        pressure = c2.checkbox("Under pressure")
        one_on_one = c1.checkbox("One-on-one")
        open_goal = c2.checkbox("Open goal")
        n_cone = st.slider("Defenders between the ball and the goal", 0, 6, 0)
        n_close = st.slider("Defenders within 3 yards of the shooter", 0, 5, 0)
        gk_dist = st.slider("Keeper distance from goal (yards)", 0, 40, 2)

    x, y = 120 - depth, 40 + lateral
    flags = {
        "is_header": int(body == "Head"),
        "under_pressure": int(pressure),
        "shot_first_time": int(first_time),
        "shot_one_on_one": int(one_on_one),
        "shot_open_goal": int(open_goal),
        "n_def_in_cone": n_cone,
        "n_def_close": n_close,
        "gk_dist_to_goal": gk_dist,
    }
    xg = predict_xg(model, x, y, flags)

    with view:
        r1, r2 = st.columns([1, 1.6])
        r1.metric("Expected goals", f"{xg:.2f}")
        r2.markdown(f'<div class="verdict">{describe_chance(xg)}</div>', unsafe_allow_html=True)
        r2.caption(f"Scored about once every {max(1, round(1 / xg))} attempts like this.")

        fig = draw_simulator_pitch(xg_surface(model, flags), x, y)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("The heatmap shows the xG of this same type of shot from every spot: "
                   "warmer means more likely to go in.")

        nearby = shots[np.hypot(shots["x"] - x, shots["y"] - y) <= 3]
        if len(nearby) >= 5:
            note = (f"In the real data, {len(nearby)} shots were taken within 3 yards of this "
                    f"spot and {nearby['is_goal'].mean():.0%} of them were scored.")
        else:
            note = "Very few real shots were taken from this spot, so treat this estimate with care."
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- shot maps
with tab_map:
    f1, f2, f3 = st.columns([1, 2, 2])
    view_by = f1.radio("View by", ["Player", "Team"])
    competitions = f2.multiselect("Tournaments", sorted(shots["competition"].unique()),
                                  default=sorted(shots["competition"].unique()))
    pool = shots[shots["competition"].isin(competitions)]

    if pool.empty:
        st.warning("Select at least one tournament to see shots.")
    else:
        column = "player" if view_by == "Player" else "team"
        default_name = "Kylian Mbappé Lottin" if view_by == "Player" else "France"
        options = sorted(pool[column].unique())
        index = options.index(default_name) if default_name in options else 0
        name = f3.selectbox(f"Choose a {column}", options, index=index)
        selection = pool[pool[column] == name]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Shots", len(selection))
        k2.metric("Goals", int(selection["is_goal"].sum()))
        k3.metric("xG", f"{selection['xg'].sum():.2f}")
        k4.metric("Goals minus xG", f"{selection['is_goal'].sum() - selection['xg'].sum():+.2f}")

        left, right = st.columns([1.5, 1], gap="large")
        with left:
            fig = draw_shot_map(selection, name)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Circle size is the xG of the shot. Penalties are excluded.")
        with right:
            table_cols = {"competition": "Tournament", "minute": "Minute", "xg": "xG"}
            if view_by == "Team":
                table_cols = {"player": "Player", **table_cols}
            xg_format = {"xG": st.column_config.NumberColumn(format="%.2f")}
            st.markdown("#### Hardest goals scored")
            hardest = selection[selection["is_goal"] == 1].nsmallest(5, "xg")
            if hardest.empty:
                st.caption("No goals in this selection.")
            else:
                st.dataframe(hardest[list(table_cols)].rename(columns=table_cols),
                             hide_index=True, width="stretch", column_config=xg_format)
            st.markdown("#### Biggest chances missed")
            missed = selection[selection["is_goal"] == 0].nlargest(5, "xg")
            st.dataframe(missed[list(table_cols)].rename(columns=table_cols),
                         hide_index=True, width="stretch", column_config=xg_format)

# ---------------------------------------------------------------- leaderboard
with tab_board:
    b1, b2 = st.columns([2, 1])
    board_comps = b1.multiselect("Tournaments ", sorted(shots["competition"].unique()),
                                 default=sorted(shots["competition"].unique()))
    min_shots = b2.slider("Minimum shots", 5, 50, 20)

    board = (
        shots[shots["competition"].isin(board_comps)]
        .groupby("player")
        .agg(team=("team", lambda s: s.mode().iat[0]), shots=("is_goal", "size"),
             goals=("is_goal", "sum"), xg=("xg", "sum"))
    )
    board["goals_minus_xg"] = board["goals"] - board["xg"]
    board["xg_per_shot"] = board["xg"] / board["shots"]
    board = board[board["shots"] >= min_shots]

    if board.empty:
        st.warning("No player matches these filters. Lower the minimum number of shots "
                   "or add tournaments.")
    else:
        top = board["goals_minus_xg"].nlargest(10)
        bottom = board["goals_minus_xg"].nsmallest(10)
        g1, g2 = st.columns(2, gap="large")
        with g1:
            fig = draw_bars(top, GOLD, "Scored more than expected")
            st.pyplot(fig)
            plt.close(fig)
        with g2:
            fig = draw_bars(bottom, RED, "Scored less than expected")
            st.pyplot(fig)
            plt.close(fig)

        st.dataframe(
            board.sort_values("goals_minus_xg", ascending=False).reset_index(),
            hide_index=True,
            width="stretch",
            column_config={
                "player": "Player",
                "team": "Team",
                "shots": "Shots",
                "goals": "Goals",
                "xg": st.column_config.NumberColumn("xG", format="%.2f"),
                "goals_minus_xg": st.column_config.NumberColumn("Goals minus xG", format="%+.2f"),
                "xg_per_shot": st.column_config.NumberColumn("xG per shot", format="%.3f"),
            },
        )
        st.caption("Over a few dozen shots, luck plays a big part in these numbers: "
                   "a single long-range goal can add almost one full goal on its own.")

# ---------------------------------------------------------------- model
with tab_model:
    st.subheader("Holdout test on Euro 2024")
    st.write("The model was trained on the 2018 and 2022 World Cups and Euro 2020, then "
             "tested on Euro 2024, a tournament it had never seen. It captures about 83% of "
             "the improvement StatsBomb's commercial model makes over a naive baseline.")
    holdout = pd.DataFrame({
        "Model": ["Naive baseline", "Logistic regression (this project)",
                  "XGBoost, tuned", "StatsBomb xG"],
        "Log loss (lower is better)": [0.2692, 0.2324, 0.2357, 0.2251],
        "Brier score (lower is better)": [0.0699, 0.0634, 0.0641, 0.0615],
        "ROC AUC (higher is better)": [0.500, 0.767, 0.754, 0.787],
    })
    st.dataframe(holdout, hide_index=True, width="stretch", column_config={
        col: st.column_config.NumberColumn(format="%.3f") for col in holdout.columns[1:]
    })

    st.subheader("All shots, out-of-fold predictions")
    y_true = shots["is_goal"]
    candidates = {
        "This model": shots["xg"],
        "StatsBomb xG": shots["shot_statsbomb_xg"],
    }
    cal_col, coef_col = st.columns(2, gap="large")

    with cal_col:
        fig, ax = plt.subplots(figsize=(6, 5.4))
        ax.plot([0, 0.6], [0, 0.6], linestyle="--", color=SAGE, label="Perfect calibration")
        for (label, preds), color in zip(candidates.items(), [GOLD, CHALK]):
            frac, mean_pred = calibration_curve(y_true, preds, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac, "o-", color=color, label=label)
        ax.set_xlabel("Predicted xG")
        ax.set_ylabel("Share of shots scored")
        ax.set_title("Calibration", fontsize=14, pad=10)
        style_axes(fig, ax)
        ax.legend(facecolor=TURF_DARK, edgecolor=TURF_DARK, labelcolor=CHALK)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        scores = " | ".join(
            f"{label}: log loss {log_loss(y_true, p):.4f}, Brier {brier_score_loss(y_true, p):.4f}, "
            f"AUC {roc_auc_score(y_true, p):.3f}"
            for label, p in candidates.items()
        )
        st.caption(scores)

    with coef_col:
        coefs = pd.Series(model.named_steps["logisticregression"].coef_[0], index=FEATURES)
        coefs = coefs.rename(FEATURE_LABELS).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5.4))
        ax.barh(coefs.index, coefs.values, color=[RED if v < 0 else GOLD for v in coefs.values])
        ax.axvline(0, color=SAGE, linewidth=1)
        ax.set_title("What drives the xG of a shot", fontsize=14, pad=10)
        ax.grid(axis="y", visible=False)
        style_axes(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Standardised logistic regression coefficients: gold raises the chance "
                   "of scoring, red lowers it.")

    with st.expander("How it was built"):
        st.markdown(
            """
- **Data**: 5,829 shots from 230 matches, downloaded with `statsbombpy`. Penalties are
  removed, leaving 5,606 open-play and set-piece shots.
- **Features**: distance and angle to goal, body part, shot context flags, and three
  pressure features computed from StatsBomb's freeze frames: defenders inside the shooting
  cone, defenders close to the shooter, and the keeper's distance from goal.
- **Model**: scikit-learn pipeline (median imputation, standard scaling, logistic
  regression). A tuned XGBoost was tested and performed worse on the holdout.
- **Shot values in this app** are out-of-fold predictions, so no shot is scored by a model
  that saw it during training.
"""
        )

st.divider()
st.caption("Data: StatsBomb Open Data. Model and app built with pandas, scikit-learn, "
           "mplsoccer and Streamlit.")
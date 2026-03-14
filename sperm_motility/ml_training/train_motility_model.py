# ======================================================
# VISEM MOTILITY PREDICTION PIPELINE (LOCAL VERSION)
# YOLOv8 + ByteTrack + CASA Features + ExtraTrees
# ======================================================

import os
import cv2
import joblib
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from scipy.stats import skew, kurtosis

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ======================================================
# PATHS (LOCAL)
# ======================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VIDEO_DIR = os.path.join(BASE_DIR, "sperm_motility_dataset", "videos")
VIDEO_MAP_PATH = os.path.join(BASE_DIR, "sperm_motility_dataset", "videos.csv")
SEMEN_PATH = os.path.join(BASE_DIR, "sperm_motility_dataset", "semen_analysis_data.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo", "best.pt")

OUTPUT_DIR = os.path.join(BASE_DIR, "motility_output")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "motility")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ======================================================
# LOAD DATA
# ======================================================

video_df = pd.read_csv(VIDEO_MAP_PATH, sep=";")
semen_df = pd.read_csv(SEMEN_PATH, sep=";")

semen_df = semen_df[[
    "ID",
    "Progressive motility (%)",
    "Non progressive sperm motility (%)",
    "Immotile sperm (%)"
]]

semen_df.columns = ["ID", "PR", "NP", "IM"]

gt_df = video_df.merge(semen_df, on="ID")
gt_df["video_id"] = gt_df["video"].str.replace(".avi", "", regex=False)

print("Total GT videos:", len(gt_df))


# ======================================================
# LOAD YOLO MODEL
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH)
model.to(device)


# ======================================================
# FEATURE EXTRACTION
# ======================================================

all_features = []

for vid in tqdm(sorted(os.listdir(VIDEO_DIR))):

    video_path = os.path.join(VIDEO_DIR, vid)
    video_name = vid.replace(".avi", "")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    tracker_yaml = os.path.join(OUTPUT_DIR, "tracker.yaml")

    with open(tracker_yaml, "w") as f:
        f.write(f"""tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.05
new_track_thresh: 0.5
track_buffer: 200
match_thresh: 0.9
fuse_score: True
proximity_thresh: 0.8
appearance_thresh: 0.25
frame_rate: {fps}
""")

    results = model.track(
        source=video_path,
        tracker=tracker_yaml,
        conf=0.3,
        stream=True,
        persist=True,
        device=device,
        verbose=False
    )

    tracking_data = []

    for frame_idx, r in enumerate(results):

        if r.boxes.id is None:
            continue

        ids = r.boxes.id.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for i in range(len(ids)):
            x1, y1, x2, y2 = boxes[i]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            tracking_data.append([frame_idx, int(ids[i]), cx, cy])

    if len(tracking_data) == 0:
        continue

    df = pd.DataFrame(tracking_data, columns=["frame", "id", "x", "y"])

    min_frames = int(0.5 * fps)

    features = []

    for sperm_id, g in df.groupby("id"):

        g = g.sort_values("frame")

        if len(g) < min_frames:
            continue

        duration = len(g) / fps

        dx = g["x"].iloc[-1] - g["x"].iloc[0]
        dy = g["y"].iloc[-1] - g["y"].iloc[0]

        disp = np.sqrt(dx**2 + dy**2)

        VSL = disp / duration if duration > 0 else 0

        dx_step = np.diff(g["x"])
        dy_step = np.diff(g["y"])

        step = np.sqrt(dx_step**2 + dy_step**2)

        path_length = step.sum()

        VCL = path_length / duration if duration > 0 else 0

        LIN = VSL / VCL if VCL > 0 else 0

        accel = np.mean(np.abs(np.diff(step))) if len(step) > 2 else 0

        features.append([VSL, VCL, LIN, disp, duration, accel])

    if len(features) == 0:
        continue

    vel_df = pd.DataFrame(
        features,
        columns=["VSL", "VCL", "LIN", "disp", "duration", "accel"]
    )

    video_features = {

        "video_id": video_name,

        "mean_VSL": vel_df["VSL"].mean(),
        "median_VSL": vel_df["VSL"].median(),
        "std_VSL": vel_df["VSL"].std(),

        "skew_VSL": skew(vel_df["VSL"]),
        "kurt_VSL": kurtosis(vel_df["VSL"]),

        "p10_VSL": vel_df["VSL"].quantile(.10),
        "p25_VSL": vel_df["VSL"].quantile(.25),
        "p50_VSL": vel_df["VSL"].quantile(.50),
        "p75_VSL": vel_df["VSL"].quantile(.75),
        "p90_VSL": vel_df["VSL"].quantile(.90),

        "mean_VCL": vel_df["VCL"].mean(),
        "median_VCL": vel_df["VCL"].median(),

        "mean_LIN": vel_df["LIN"].mean(),
        "median_LIN": vel_df["LIN"].median(),

        "mean_accel": vel_df["accel"].mean(),

        "track_count": len(vel_df),

        "mean_duration": vel_df["duration"].mean()
    }

    all_features.append(video_features)


# ======================================================
# BUILD DATASET
# ======================================================

feature_df = pd.DataFrame(all_features)

final_df = feature_df.merge(
    gt_df[["video_id", "PR", "NP", "IM"]],
    on="video_id"
)

print("Final dataset size:", len(final_df))


# SAVE FEATURES
feature_df.to_csv(os.path.join(OUTPUT_DIR, "extracted_motion_features.csv"), index=False)


# ======================================================
# MACHINE LEARNING
# ======================================================

feature_cols = [c for c in final_df.columns if c not in ["video_id", "PR", "NP", "IM"]]

X = final_df[feature_cols].values
y = final_df[["PR", "NP"]].values

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

best_mae = 999

overall = []
pr_scores = []
np_scores = []
im_scores = []

for train_idx, test_idx in rkf.split(X):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred = np.clip(pred, 0, 100)

    sum_two = pred[:, 0] + pred[:, 1]
    mask = sum_two > 100
    pred[mask] = pred[mask] / sum_two[mask][:, None] * 100

    IM = 100 - pred[:, 0] - pred[:, 1]
    IM = np.clip(IM, 0, 100)

    pred_full = np.column_stack([pred, IM])

    gt = final_df[["PR", "NP", "IM"]].iloc[test_idx].values

    mae = mean_absolute_error(gt, pred_full)

    overall.append(mae)

    pr_scores.append(mean_absolute_error(gt[:, 0], pred_full[:, 0]))
    np_scores.append(mean_absolute_error(gt[:, 1], pred_full[:, 1]))
    im_scores.append(mean_absolute_error(gt[:, 2], pred_full[:, 2]))

    if mae < best_mae:

        best_mae = mae

        joblib.dump(model, os.path.join(MODEL_SAVE_DIR, "best_motility_model.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "feature_scaler.pkl"))


print("\n===== FINAL RESULTS =====")

print("Overall MAE:", np.mean(overall))
print("PR MAE:", np.mean(pr_scores))
print("NP MAE:", np.mean(np_scores))
print("IM MAE:", np.mean(im_scores))

print("\nBest model saved")
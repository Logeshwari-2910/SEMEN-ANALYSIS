import os
import cv2
import json
import joblib
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.stats import skew, kurtosis

# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VIDEO_PATH = os.path.join(BASE_DIR, "test_video.avi")

YOLO_MODEL = os.path.join(BASE_DIR, "models/yolo/best.pt")
ML_MODEL = os.path.join(BASE_DIR, "models/motility/best_motility_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/motility/feature_scaler.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "motility_output")

TRACK_VIDEO_DIR = os.path.join(OUTPUT_DIR, "tracked_videos")
CASA_DIR = os.path.join(OUTPUT_DIR, "casa_features")
TRAJ_DIR = os.path.join(OUTPUT_DIR, "trajectories")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(TRACK_VIDEO_DIR, exist_ok=True)
os.makedirs(CASA_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]


# ==============================
# LOAD MODELS
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"

detector = YOLO(YOLO_MODEL)
detector.to(device)

ml_model = joblib.load(ML_MODEL)
scaler = joblib.load(SCALER_PATH)

print("Models loaded successfully")


# ==============================
# VIDEO INFO
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()


# ==============================
# CREATE TRACKER CONFIG
# ==============================

tracker_yaml = os.path.join(OUTPUT_DIR, "tracker.yaml")

with open(tracker_yaml,"w") as f:
    f.write(f"""
tracker_type: bytetrack
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


# ==============================
# RUN TRACKING
# ==============================

results = detector.track(
    source=VIDEO_PATH,
    tracker=tracker_yaml,
    conf=0.3,
    stream=True,
    persist=True,
    save=True,
    project=TRACK_VIDEO_DIR,
    name=video_name,
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

        x1,y1,x2,y2 = boxes[i]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        tracking_data.append([frame_idx,int(ids[i]),cx,cy])


df = pd.DataFrame(tracking_data,columns=["frame","id","x","y"])


# ==============================
# SAVE TRAJECTORIES
# ==============================

traj_path = os.path.join(TRAJ_DIR,f"{video_name}_tracks.csv")
df.to_csv(traj_path,index=False)

print("Trajectory saved:",traj_path)


# ==============================
# CASA FEATURE EXTRACTION
# ==============================

min_frames = int(0.5 * fps)

features = []

for sperm_id,g in df.groupby("id"):

    g = g.sort_values("frame")

    if len(g) < min_frames:
        continue

    duration = len(g)/fps

    dx = g["x"].iloc[-1] - g["x"].iloc[0]
    dy = g["y"].iloc[-1] - g["y"].iloc[0]

    disp = np.sqrt(dx**2 + dy**2)

    VSL = disp/duration if duration>0 else 0

    dx_step = np.diff(g["x"])
    dy_step = np.diff(g["y"])

    step = np.sqrt(dx_step**2 + dy_step**2)

    path_length = step.sum()

    VCL = path_length/duration if duration>0 else 0

    LIN = VSL/VCL if VCL>0 else 0

    accel = np.mean(np.abs(np.diff(step))) if len(step)>2 else 0

    features.append([VSL,VCL,LIN,disp,duration,accel])


vel_df = pd.DataFrame(
    features,
    columns=["VSL","VCL","LIN","disp","duration","accel"]
)


# ==============================
# SAVE CASA FEATURES
# ==============================

casa_path = os.path.join(CASA_DIR,f"{video_name}_features.csv")
vel_df.to_csv(casa_path,index=False)

print("CASA features saved:",casa_path)


# ==============================
# AGGREGATE VIDEO FEATURES
# ==============================

video_features = {

"mean_VSL":vel_df["VSL"].mean(),
"median_VSL":vel_df["VSL"].median(),
"std_VSL":vel_df["VSL"].std(),

"skew_VSL":skew(vel_df["VSL"]),
"kurt_VSL":kurtosis(vel_df["VSL"]),

"p10_VSL":vel_df["VSL"].quantile(.10),
"p25_VSL":vel_df["VSL"].quantile(.25),
"p50_VSL":vel_df["VSL"].quantile(.50),
"p75_VSL":vel_df["VSL"].quantile(.75),
"p90_VSL":vel_df["VSL"].quantile(.90),

"mean_VCL":vel_df["VCL"].mean(),
"median_VCL":vel_df["VCL"].median(),

"mean_LIN":vel_df["LIN"].mean(),
"median_LIN":vel_df["LIN"].median(),

"mean_accel":vel_df["accel"].mean(),

"track_count":len(vel_df),

"mean_duration":vel_df["duration"].mean()
}


X = pd.DataFrame([video_features])

X_scaled = scaler.transform(X)


# ==============================
# PREDICT MOTILITY
# ==============================

pred = ml_model.predict(X_scaled)[0]

PR = max(0,min(100,pred[0]))
NP = max(0,min(100,pred[1]))

if PR+NP>100:
    PR = PR/(PR+NP)*100
    NP = NP/(PR+NP)*100

IM = 100-PR-NP


print("\n===== MOTILITY RESULT =====")

print(f"Progressive (PR): {PR:.2f}%")
print(f"Non Progressive (NP): {NP:.2f}%")
print(f"Immotile (IM): {IM:.2f}%")


# ==============================
# SAVE PREDICTION
# ==============================

result = {
    "video": video_name,
    "progressive_motility": float(PR),
    "non_progressive_motility": float(NP),
    "immotile": float(IM)
}

result_path = os.path.join(PRED_DIR,f"{video_name}_result.json")

with open(result_path,"w") as f:
    json.dump(result,f,indent=4)

print("Prediction saved:",result_path)
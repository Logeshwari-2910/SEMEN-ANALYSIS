# Sperm Motility

This section explains how to use the **`sperm_motility_project`** folder.

---

# Project Structure

```
sperm_motility_project
│
├── data.yaml
├── requirements.txt
├── yolo_training
│   └── train_yolo.py
├── ml_training
│   └── train_motility_model.py
├── motility_inference
│   └── predict_motility.py
├── models
│   ├── yolo
│   │   └── best.pt
│   └── motility
│       ├── best_motility_model.pkl
│       └── feature_scaler.pkl
├── sperm_motility_dataset
│   ├── videos
│   ├── videos.csv
│   └── semen_analysis_data.csv
└── motility_output
    ├── tracked_videos
    ├── casa_features
    ├── trajectories
    └── predictions
```

---

# Folder Description

**data.yaml**
Configuration file used for YOLO training.

**requirements.txt**
Contains the required Python libraries.

**yolo_training/**
Contains the script used to train the sperm detection model.

**ml_training/**
Contains the script used to train the motility prediction model using extracted motion features.

**motility_inference/**
Contains the script used to run inference and predict sperm motility from a video.

**models/**
Stores trained models:

* YOLOv8 detection model
* Motility prediction model
* Feature scaler

**sperm_motility_dataset/**
Contains dataset files used during training.

**motility_output/**
Stores outputs generated during training and inference.

---

# Dataset

This project uses the **VISEM dataset**.

Dataset link:
https://www.kaggle.com/datasets/stevenhicks/visem-video-dataset

The repository already contains:

* `videos.csv`
* `semen_analysis_data.csv`

These files provide the mapping between video names and the ground truth motility values.

---

## For Training

To train the system, download the **VISEM videos** and place them inside:

```
sperm_motility_dataset/videos/
```

Example:

```
sperm_motility_dataset/
│
├── videos/
│   ├── 7_12.avi
│   ├── 11_09.avi
│   ├── 27_09.avi
│   └── ...
│
├── videos.csv
└── semen_analysis_data.csv
```

---

# Installation

Install the required libraries:

```
pip install -r requirements.txt
```

---

# Training

### 1. Train YOLOv8 Detection Model

Run:

```
python yolo_training/train_yolo.py
```

After training, the YOLOv8 model will be saved in:

```
runs/detect/train/weights/best.pt
```

Copy this file to:

```
models/yolo/best.pt
```

---

### 2. Train Motility Prediction Model

Run:

```
python ml_training/train_motility_model.py
```

During training:

* sperm videos are processed
* sperm trajectories are extracted
* CASA motion features are computed
* features from **all videos (~85 VISEM videos)** are extracted

The extracted motion features will be stored in:

```
motility_output/casa_features/
```

The trained motility prediction model will be saved in:

```
models/motility/
```

---

# Inference / Testing

If you only want to run inference, you do **not need to run training**.

1. Download any video from the **VISEM dataset**.
2. Place the video anywhere in the project folder.

Open the file:

```
motility_inference/predict_motility.py
```

Change the input video path:

```
VIDEO_PATH = "test_video.avi"
```

Then run:

```
python motility_inference/predict_motility.py
```

---

# Output

Results will be stored in:

```
motility_output/
```

The folder may contain:

* tracked videos
* extracted CASA motion features
* sperm trajectory data
* predicted motility values

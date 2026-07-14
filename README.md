# COE-486 Computer Vision — Semester Project
## Attention-Augmented YOLOv8 for Improved Small-Scale Vehicle Damage Detection

**American University of Sharjah — Department of Computer Engineering**  
**Instructor:** Dr. Omar Arif  
**Group Members:** Abdulaziz Alsuhaimi, Ali AlQahtani, Mohamed AlZarooni, Jad Freij

---

## Project Overview

This project extends the YOLOv8s object detection framework by integrating a Convolutional Block Attention Module (CBAM) into the backbone to improve detection of small-scale vehicle damage. We train and evaluate a baseline YOLOv8s model and a CBAM-augmented variant on the CarDD dataset, which covers six damage categories: dent, scratch, crack, glass shatter, lamp broken, and tire flat.

The unified CBAM model replaces three previously trained single-class specialist models, detecting all six damage types in a single inference pass.

---

## Dataset

We use the **CarDD (Car Damage Detection)** dataset, available on Kaggle:  
https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels

- ~4,000 high-resolution images
- 6 damage categories
- Pre-split into train / val / test sets
- YOLO format annotations

---

## Requirements

- Google Colab (recommended) with GPU enabled: Runtime → Change runtime type → T4 GPU
- Kaggle account + API token (for dataset download)
- Python 3.10+
- ultralytics >= 8.0
- torch >= 2.0
- PyYAML, matplotlib, pandas

---

## How to Run

### Step 1 — Open the Notebook
Open `CV_ProjectCode.ipynb` in Google Colab:  
**File → Open notebook → GitHub → paste this repo URL**

Or open directly via the Colab link at the top of the notebook file.

### Step 2 — Enable GPU
In Colab: **Runtime → Change runtime type → T4 GPU → Save**

### Step 3 — Run Cell 1: Install Dependencies
```python
!pip install ultralytics -q
```
This installs YOLOv8 and all required libraries.

### Step 4 — Run Cell 2: Download Dataset
Fill in your Kaggle credentials:
```python
kaggle_token = "YOUR_KAGGLE_TOKEN_HERE"
# username: "YOUR_KAGGLE_USERNAME_HERE"
```
Then run the cell. The CarDD dataset (~2.8GB) will be downloaded and extracted automatically.

> To get your Kaggle token: kaggle.com → Settings → API → Create New Token

### Step 5 — Run Cell 3: Fix Dataset Paths
This updates the data.yaml file to point to the correct Colab paths. Run as-is, no changes needed.

### Step 6 (Training) — Run Cell 4: Train Baseline YOLOv8s
```python
baseline_model = YOLO('yolov8s.pt')
baseline_model.train(data='/content/cardd/data.yaml', epochs=10, ...)
```
Training takes approximately 15-20 minutes on a T4 GPU.

### Step 7 — Run Cell 5: Check Baseline Results
Prints mAP@0.5, mAP@0.5:0.95, Precision, and Recall for the baseline model.

### Step 8 (Training) — Run Cell 6: Train CBAM-Augmented Model
Defines the CBAM attention module, injects it into the YOLOv8s backbone, and trains the augmented model. Takes approximately 15-20 minutes.

### Step 9 — Run Cell 7: Compare Results
Prints a side-by-side comparison table and generates a bar chart saved to `/content/comparison_chart.png`.

### Step 10 (Testing) — Run Cell 8: Test Specialist Models
Upload the three specialist model files when prompted:
- `best_window.pt` — broken glass detector
- `best_lamp.pt` — lamp damage detector  
- `best_damage.pt` — crack and dent detector

The cell runs each model on sample test images and generates a visual comparison grid.

---

## Testing Only (Skip Training)

If you want to test the trained CBAM model without retraining:

1. Download `best.pt` from the project Google Drive (link provided separately)
2. Run this in a new Colab cell:

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('your_car_image.jpg', conf=0.25)
results[0].show()
```

This runs the unified model on any car image and draws bounding boxes around detected damage.

---

## Results Summary

> **Note:** CBAM is currently injected as a wrapper around backbone layer 4. 
> Future work will embed it directly within the C2f block for full fusion compatibility.

| Metric | Baseline YOLOv8s | YOLOv8s + CBAM | Improvement |
|---|---|---|---|
| mAP@0.5 | 0.427 | 0.598 | ▲ 40.0% |
| mAP@0.5:0.95 | 0.272 | 0.469 | ▲ 72.2% |
| Precision | 0.434 | 0.635 | ▲ 46.2% |
| Recall | 0.431 | 0.566 | ▲ 31.4% |

### Per-Class mAP@0.5 (CBAM Model)

| Class | mAP@0.5 |
|---|---|
| dent | 0.420 |
| scratch | 0.437 |
| crack | 0.096 |
| glass shatter | 0.974 |
| lamp broken | 0.751 |
| tire flat | 0.906 |

---

## Repository Structure

```
COE486-Vehicle-Damage-Detection/
│
├── CV_ProjectCode.ipynb       # Main notebook (all cells)
├── README.md                  # This file
```

---

## References

1. Redmon et al., "You Only Look Once," CVPR 2016
2. Ultralytics, "YOLOv8," https://github.com/ultralytics/ultralytics
3. Lin et al., "Feature Pyramid Networks," CVPR 2017
4. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
5. Ma, "YOLOv5-CBAM: A Small Object Detection Model," IEEE RICAI 2024
6. Agrawal et al., "Advanced Car Damage Assessment Using YOLOv8," ResearchGate 2025
7. Wang et al., "CarDD: A New Dataset for Vision-Based Car Damage Detection," IEEE Trans. ITS 2023

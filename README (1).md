# COE-486 Computer Vision — Semester Project
## Attention-Augmented YOLOv8 for Improved Small-Scale Vehicle Damage Detection

**American University of Sharjah — Department of Computer Engineering**  
**Instructor:** Dr. Omar Arif  
**Group Members:** Abdulaziz Alsuhaimi, Ali AlQahtani, Mohamed AlZarooni, Jad Freij

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abdulaziz-Alsuhaimi/COE486-Vehicle-Damage-Detection/blob/main/CV_ProjectCode.ipynb)

---

## Project Overview

This project extends the YOLOv8s object detection framework by integrating a Convolutional Block Attention Module (CBAM) into the backbone to improve detection of small-scale vehicle damage. We train and evaluate a baseline YOLOv8s model and a CBAM-augmented variant on the CarDD dataset, which covers six damage categories: dent, scratch, crack, glass shatter, lamp broken, and tire flat.

The unified CBAM model replaces three previously trained single-class specialist models, detecting all six damage types in a single inference pass.

### Implementation note — how CBAM is injected

`YOLO.train()` rebuilds the model from its yaml config and re-loads weights by
name, so modules attached directly to `model.model` before training are
silently discarded. We therefore inject CBAM inside a custom
`DetectionTrainer.get_model()` override (Cell 6), which runs *after* the
rebuilt model has loaded its pretrained weights. The training log confirms the
injection: it prints `CBAM injected after backbone layer 4 (+2,146 parameters)`
and shows the same `Transferred 349/355 items` as the baseline run (no weights
lost to renaming).

---

## Dataset

We use the **CarDD (Car Damage Detection)** dataset, available on Kaggle:  
https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels

- ~4,000 high-resolution images
- 6 damage categories
- Pre-split into train / val / test sets
- YOLO format annotations

The notebook downloads it with `kagglehub` — the dataset is public, so **no
Kaggle account or API token is needed**.

---

## Requirements

- Google Colab (recommended) with GPU enabled: Runtime → Change runtime type → T4 GPU
- Python 3.10+
- ultralytics >= 8.0 and kagglehub (both installed by Cell 1; torch, PyYAML, and matplotlib come with the Colab runtime / ultralytics)

---

## How to Run

### Step 1 — Open the Notebook
Click the Colab badge at the top of this README, or in Colab:
**File → Open notebook → GitHub → paste this repo URL**

### Step 2 — Enable GPU
In Colab: **Runtime → Change runtime type → T4 GPU → Save**

### Step 3 — Run Cell 1: Install Dependencies
Installs `ultralytics` and `kagglehub`, and prints the environment check.

### Step 4 — Run Cell 2: Download Dataset
Downloads the CarDD dataset (~2.8 GB) via `kagglehub`. No credentials needed.

### Step 5 — Run Cell 3: Fix Dataset Paths
Points `data.yaml` at the downloaded copy and verifies the train/val/test
folders exist. Run as-is, no changes needed.

### Step 6 (Training) — Run Cell 4: Train Baseline YOLOv8s
Trains the baseline for 10 epochs (~15–20 minutes on a T4 GPU).

### Step 7 — Run Cell 5: Evaluate Baseline
Evaluates `best.pt` on the held-out **test** split and stores the metrics for
the comparison in Cell 9.

### Step 8 — Run Cell 6: Define CBAM + Custom Trainer
Defines the CBAM attention modules and the `CBAMTrainer` that injects CBAM
into the backbone in a way that survives training (see Implementation note).

### Step 9 (Training) — Run Cell 7: Train CBAM-Augmented Model
Trains YOLOv8s + CBAM with identical settings (~15–20 minutes). **Check the
log for the `CBAM injected after backbone layer 4` line** — that is the proof
the attention module is actually in the trained network.

### Step 10 — Run Cell 8: Evaluate CBAM Model
Evaluates the CBAM `best.pt` on the same test split, including per-class
mAP@0.5.

### Step 11 — Run Cell 9: Compare Results
Prints a side-by-side comparison computed from the two runs (nothing
hardcoded), a markdown table ready to paste into this README, and saves a
bar chart to `/content/comparison_chart.png`.

### Step 12 (Optional) — Run Cell 10: Test Specialist Models
Download the three legacy specialist models from the
[project Google Drive](https://drive.google.com/drive/folders/15DkkoSd_iVtZUCvcQjyWwp5BiO06iGAp)
and upload them to the Colab session (file panel on the left):
- `best_window.pt` — broken glass detector
- `best_lamp.pt` — lamp damage detector
- `best_damage.pt` — crack and dent detector

The cell runs whichever models are present on sample test images and
generates a visual comparison grid.

---

## Testing Only (Skip Training)

If you want to test the trained unified model without retraining, the weights
live in the [project Google Drive](https://drive.google.com/drive/folders/15DkkoSd_iVtZUCvcQjyWwp5BiO06iGAp)
(`cbam_yolov8s/weights/best.pt`).

> **⚠️ Important:** the `best.pt` currently on Drive was trained **before**
> the CBAM-injection fix — it is effectively a stock YOLOv8s, not a CBAM
> model (see the Results note below). It still detects all six damage
> classes, but it does not demonstrate the attention mechanism. After
> re-running Cells 4–9, replace it with the new
> `/content/runs/cbam_yolov8s/weights/best.pt`.

1. Download `best.pt` from the Drive folder (`cbam_yolov8s/weights/best.pt`)
   and upload it to the Colab session (file panel on the left), along with a
   car image to test on.
2. Run **Cell 1** (install) and **Cell 6** (CBAM class definitions — required
   for checkpoints trained after the fix; harmless for the old one).
3. Paste this into a new cell and run it:

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('your_car_image.jpg', conf=0.25)
results[0].show()
```

This runs the unified model on any car image and draws bounding boxes around detected damage.

---

## Results

> **Note:** Results reported by earlier versions of this repository were
> invalid: the CBAM injection did not survive `YOLO.train()`'s internal model
> rebuild (the "CBAM" run was actually a stock YOLOv8s with one backbone stage
> randomly re-initialized), and the comparison cell mixed hardcoded numbers
> from different sessions. The tables below are from a full re-run with the
> fixed injection (RTX 5070 Ti, 10 epochs, seed 0, evaluated on the test
> split; the committed notebook contains the complete logs).

| Metric | Baseline YOLOv8s | YOLOv8s + CBAM | Change |
|---|---|---|---|
| mAP@0.5 | 0.664 | 0.667 | ▲ 0.4% |
| mAP@0.5:0.95 | 0.511 | 0.510 | ▼ 0.1% |
| Precision | 0.652 | 0.718 | ▲ 10.0% |
| Recall | 0.625 | 0.609 | ▼ 2.5% |

### Per-Class mAP@0.5 (CBAM model, test split)

| Class | mAP@0.5 |
|---|---|
| dent | 0.542 |
| scratch | 0.524 |
| crack | 0.264 |
| glass shatter | 0.984 |
| lamp broken | 0.779 |
| tire flat | 0.906 |

**Discussion.** At 10 epochs the CBAM variant is at parity with the baseline
on overall mAP (+0.4% mAP@0.5, −0.1% mAP@0.5:0.95). We repeated the identical
configuration on two machines (Colab T4 and a local RTX 5070 Ti, both
seed 0): overall mAP was at parity both times (0.681→0.684 and 0.664→0.667),
but the precision/recall balance flipped between runs (T4: precision −11.7%,
recall +7.6%; local: precision +10.0%, recall −2.5%), and per-class results
on the rare `crack` class swung in opposite directions. The honest
conclusion is that at this training budget, CBAM neither helps nor hurts
overall accuracy, and run-to-run variance dominates any per-class effect —
longer training and multiple seeds would be needed to detect a real
difference.

---

## Limitations

- 10 epochs with a single seed — results are indicative, not definitive;
  run-to-run variance was not measured.
- Metrics are reported on the CarDD **test** split (earlier versions reported
  validation-split metrics from the training loop).
- `crack` is by far the hardest class for both models (small, low-contrast
  damage regions).
- CBAM placement was only evaluated after backbone layer 4; other placements
  and multiple insertion points were not explored.

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

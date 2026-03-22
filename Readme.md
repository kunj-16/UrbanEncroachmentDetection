# Urban Encroachment Detection using Multi-Temporal Satellite Imagery

This project implements an **end-to-end weakly supervised deep learning pipeline** to detect urban expansion (potential encroachment) using freely available Sentinel-2 satellite imagery.

The system compares two time periods (2019 and 2024) over a selected urban region (Lucknow, ~50 km²) and produces a **city-scale encroachment risk heatmap**.

---

## 1. Project Overview

### Problem

Manual monitoring of urban encroachment is slow and non-scalable. Simple vegetation-based methods (e.g., NDVI thresholding) produce many false positives due to seasonal and agricultural changes.

### Proposed Solution

A **CNN-based change detection framework** trained using **weak supervision** derived from NDVI change, which learns **construction-specific spatio-spectral patterns** rather than raw vegetation loss.

### Key Characteristics

* No manual ground-truth labels
* Uses only free Sentinel-2 data
* City-scale deployment
* Interpretable visual outputs

---

## 2. Folder Structure (Canonical)

```
terrain_analyzer/
│
├── data/
│   ├── raw/
│   │   ├── before_2019.tif          # Sentinel-2 composite (B02,B03,B04,B08)
│   │   └── after_2024.tif           # Sentinel-2 composite (same AOI)
│   │
│   ├── processed/
│   │   ├── urban_encroachment_stack.tif
│   │   └── encroachment_labels.tif
│   │
│   └── dataset/
│       ├── images/                  # 11×256×256 numpy patches
│       └── masks/                   # 256×256 binary masks
│
├── preprocessing/
│   ├── build_stack.py
│   ├── generate_labels.py
│   └── extract_patches.py
│
├── models/
│   ├── train_unet.py
│   ├── infer_city.py
│   └── encroachment_model.pt
│
├── evaluation/
│   ├── ndvi_baseline.py
│   └── compare_ndvi_vs_cnn.py
│
├── visualization/
│   └── visualize_stack.py
│
├── results/
│   ├── cnn_pred_map.npy
│   ├── ndvi_baseline.png
│   ├── cnn_overlay.png
│   └── comparison.png
│
└── README.md
```

---

## 3. Conceptual Pipeline

```
Raw Sentinel-2 Images
        ↓
Multi-Temporal Feature Stack (NDVI + Spectral)
        ↓
Weak Supervision (NDVI-based proxy labels)
        ↓
Patch Extraction (ML dataset)
        ↓
CNN Training (U-Net)
        ↓
City-Scale Inference
        ↓
Baseline Comparison (NDVI vs CNN)
```

Each stage produces outputs that are **consumed by the next stage**.
No file is reused ambiguously.

---

## 4. Execution Order (IMPORTANT)

### Step 0 — Raw Data (Manual)

Download the following from EO Browser (Sentinel-2 L2A, TIME RANGE mode):

* `before_2019.tif` (Jan–Mar 2019)
* `after_2024.tif` (Jan–Mar 2024)

Each file must contain **4 bands**:

```
B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
```

Place them in:

```
data/raw/
```

---

### Step 1 — Build Multi-Temporal Stack

```bash
python preprocessing/build_stack.py
```

**Input**

* data/raw/before_2019.tif
* data/raw/after_2024.tif

**Output**

* data/processed/urban_encroachment_stack.tif (11 bands)

This file is the **single source of truth** for all downstream steps.

---

### Step 2 — Generate Proxy Labels (Weak Supervision)

```bash
python preprocessing/generate_labels.py
```

**Input**

* urban_encroachment_stack.tif

**Output**

* encroachment_labels.tif

⚠️ These labels are **not ground truth**.
They are only used to guide CNN training.

---

### Step 3 — Extract Training Patches

```bash
python preprocessing/extract_patches.py
```

**Input**

* urban_encroachment_stack.tif
* encroachment_labels.tif

**Output**

```
data/dataset/images/*.npy
data/dataset/masks/*.npy
```

Each image patch has shape:

```
(11, 256, 256)
```

---

### Step 4 — Train CNN Model

```bash
python models/train_unet.py
```

**Input**

* dataset/images/
* dataset/masks/

**Output**

* models/encroachment_model.pt

This step learns **construction-like change patterns**.

---

### Step 5 — City-Scale Inference

```bash
python models/infer_city.py
```

**Input**

* urban_encroachment_stack.tif
* encroachment_model.pt

**Output**

* results/cnn_pred_map.npy
* optional visualization overlay

This produces a **continuous encroachment risk heatmap**.

---

### Step 6 — NDVI-Only Baseline

```bash
python evaluation/ndvi_baseline.py
```

**Purpose**

* Classical vegetation-loss baseline
* Used only for comparison

---

### Step 7 — CNN vs NDVI Comparison

```bash
python evaluation/compare_ndvi_vs_cnn.py
```

**Output**

* results/comparison.png

This figure demonstrates the **added value of the CNN** over NDVI thresholding.

---

## 5. Interpretation of Results

* **Red regions (CNN)**
  Areas whose multi-temporal spectral and spatial change patterns resemble construction activity.

* **NDVI baseline**
  Detects all vegetation loss (including agriculture and seasonal effects).

The CNN reduces false positives by learning **contextual and structural cues**, not just NDVI magnitude.

---

## 6. Scope and Limitations

* The model is trained on a **single city region**.
* We claim **methodological feasibility**, not nationwide generalization.
* Legal classification of encroachment is **out of scope**.

Scaling to larger regions would require **regional retraining**, not architectural changes.

---

## 7. Reproducibility Notes

* Raw data must not be modified once downloaded.
* All preprocessing scripts are deterministic.
* Results can be regenerated by re-executing the pipeline in order.

---

## 8. Intended Use

* Academic research
* Urban planning decision support
* Smart city monitoring prototypes

---

## 9. Contact / Notes

This project was developed as a **semester research project** with the goal of evolving into a publishable applied remote-sensing study.

---

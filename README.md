# Brain_Tumor_Segmentation

> **MRI-informed brain tumor segmentation using classical image processing and a geometry-regularized 3D U-Net**

This project was developed as the final project for the **MRI** course at Ben-Gurion University of the Negev.

## Overview

Automatic brain tumor segmentation is important for tumor-volume estimation, disease monitoring, treatment planning, and surgical assessment.

This project performs **binary whole-tumor segmentation** using the BraTS 2020 dataset. Each patient contains four co-registered MRI modalities:

- T1
- T1 contrast-enhanced (T1ce)
- T2
- FLAIR

We compare three approaches:

1. **Classical MRI-informed baseline** based on multimodal intensity thresholding, morphological operations, and connected-component filtering.
2. **Standard 3D U-Net** trained using Dice and binary cross-entropy losses.
3. **Geometry-regularized 3D U-Net** trained with additional Total Variation and Laplacian smoothness terms.

The main research question is:

> Can geometric regularization improve the spatial coherence of predicted tumor masks without reducing segmentation accuracy?

---

## Dataset

The project uses the **BraTS 2020** dataset, which contains multimodal brain MRI volumes and expert tumor annotations.

For binary whole-tumor segmentation, all nonzero tumor labels are mapped to foreground:

```text
Background: 0
Whole tumor: 1
````

The dataset is not included in this repository.

### Expected patient structure

```text
BraTS2020_TrainingData/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   ├── BraTS20_Training_001_flair.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
└── ...
```

---

## Methods

### Classical baseline

The classical segmentation pipeline uses MRI-specific intensity and spatial information:

```text
Brain masking
      ↓
Modality-wise normalization
      ↓
FLAIR and T2 thresholding
      ↓
T1ce support
      ↓
3D morphological operations
      ↓
Connected-component filtering
      ↓
Binary whole-tumor mask
```

FLAIR and T2 provide information about abnormal fluid-sensitive regions, while T1ce contributes information about enhancing tumor tissue.

### Standard 3D U-Net

The deep-learning model receives a four-channel 3D MRI patch:

```text
Input:  [T1, T1ce, T2, FLAIR]
Output: Whole-tumor probability map
```

The standard training objective is:

```math
\mathcal{L}_{\mathrm{standard}}
=
\mathcal{L}_{\mathrm{Dice}}
+
\lambda_{\mathrm{BCE}}\mathcal{L}_{\mathrm{BCE}}
```

### Geometry-Regularized 3D U-Net

The proposed model uses the same architecture and training data as the standard U-Net, but adds spatial regularization:

```math
\mathcal{L}_{\mathrm{geo}}
=
\mathcal{L}_{\mathrm{standard}}
+
\lambda_{\mathrm{TV}}\mathcal{L}_{\mathrm{TV}}
+
\lambda_{\mathrm{Lap}}\mathcal{L}_{\mathrm{Lap}}
```

The regularization terms encourage:

- Local spatial coherence
- Fewer isolated predictions
- Reduced boundary oscillations
- Smoother probability maps

Because real gliomas may have irregular or multifocal shapes, these terms are treated as soft constraints rather than fixed anatomical assumptions.

---

## Preprocessing

The preprocessing pipeline includes:

* Patient-level train, validation, and test splitting
* Binary whole-tumor label conversion
* Nonzero brain-mask extraction
* Modality-wise z-score normalization inside the brain mask
* Foreground-aware 3D patch sampling
* Light spatial and intensity augmentation

Patient-level splitting prevents patches from the same patient from appearing in different data splits.

---

## Evaluation

Performance is evaluated per test patient using:

* Dice similarity coefficient
* Intersection over Union (IoU)

The project reports:

* Mean and standard deviation
* Dice and IoU box plots
* Baseline-versus-model scatter plots
* Pearson correlation coefficients
* Qualitative segmentation examples
* Failure-case analysis

Secondary geometric diagnostics may include:

* Number of predicted connected components
* Surface-to-volume ratio

---

## Repository Structure

```text
Brain_Tumor_Segmentation/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── splits.py
│   ├── baselines/
│   │   └── classical_segmentation.py
│   ├── models/
│   │   └── unet3d.py
│   ├── losses/
│   │   └── geometric_losses.py
│   ├── training/
│   │   ├── train.py
│   │   └── inference.py
│   └── evaluation/
│       ├── metrics.py
│       ├── visualization.py
│       └── plots.py
├── scripts/
│   ├── run_baseline.py
│   ├── train_model.py
│   └── evaluate.py
└── outputs/
    └── .gitkeep
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<USERNAME>/Brain_Tumor_Segmentation.git
cd Brain_Tumor_Segmentation
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

The exact commands will be finalized alongside the implementation.

### Run the classical baseline

```bash
python scripts/run_baseline.py --config configs/experiment.yaml
```

### Train the standard 3D U-Net

```bash
python scripts/train_model.py \
    --config configs/experiment.yaml \
    --loss standard
```

### Train the geometry-regularized model

```bash
python scripts/train_model.py \
    --config configs/experiment.yaml \
    --loss geometric
```

### Evaluate the models

```bash
python scripts/evaluate.py --config configs/experiment.yaml
```

---

## Reproducibility

To support reproducibility:

* All paths are relative.
* Random seeds are fixed.
* Data splits are stored explicitly.
* Experiment parameters are stored in configuration files.
* The same data and architecture are used in both U-Net experiments.
* Hyperparameters are selected using only the validation set.
* The test set is evaluated after all model choices are fixed.

---

## Results

Results will be added after completing the experiments.

| Method                   | Dice Mean | Dice SD | IoU Mean | IoU SD |
| ------------------------ | --------: | ------: | -------: | -----: |
| Classical baseline       |       TBD |     TBD |      TBD |    TBD |
| 3D U-Net: Dice + BCE     |       TBD |     TBD |      TBD |    TBD |
| 3D U-Net: geometric loss |       TBD |     TBD |      TBD |    TBD |

---

## Authors

* **Yuval Ratzabi**
* **Etamar Rothstein**

Ben-Gurion University of the Negev
MRI – Final Project

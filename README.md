# Brain Tumor Segmentation from Multimodal MRI

Classical brain-tumor segmentation using the four BraTS MRI modalities: **T1, T1ce, T2, and FLAIR**.

The project compares:

* **Baseline:** a 4D Gaussian Mixture Model (GMM) using normalized MRI intensities.
* **Advanced model:** a 9D GMM combining MRI intensities, hemispheric symmetry, and distance from the brain boundary.

Both models use the same image-processing pipeline:

1. Tumor posterior estimation
2. Sobel edge detection
3. Closed-contour extraction
4. Entropy-weighted contour classification
5. Constrained region expansion
6. Binary whole-tumor segmentation

## Dataset

The project uses 369 patients from the **BraTS 2020 training dataset**. Segmentation is performed on axial slice 80.

| Split      | Patients | Purpose                  |
| ---------- | -------: | ------------------------ |
| Training   |      250 | GMM training             |
| Validation |       50 | Hyperparameter selection |
| Test       |       69 | Final evaluation         |

The dataset is not included in this repository.

## Results

| Model                   |           Mean Dice |            Mean IoU |
| ----------------------- | ------------------: | ------------------: |
| 4D Raw Baseline         |     0.6828 ± 0.3373 |     0.5998 ± 0.3231 |
| 5D Boundary Depth       |     0.7061 ± 0.3270 |     0.6234 ± 0.3123 |
| 8D Hemispheric Symmetry | **0.7323 ± 0.2769** | **0.6361 ± 0.2737** |
| 9D Boundary + Symmetry  |     0.7215 ± 0.2732 |     0.6218 ± 0.2751 |

The symmetry model achieved the highest average Dice and IoU, while the Boundary + Symmetry model produced the fewest completely missed tumors.

## Repository Structure

```text
├── main.ipynb               # Complete experimental pipeline
├── config.py                # Project configuration
├── statistical_models/      # GMM training and inference
├── image_processing/        # Contour detection and segmentation
├── evaluation/              # Optimization and evaluation
├── utilities/               # Data loading, normalization, and splitting
├── saved_parameters/        # Trained parameters
└── output/                  # Results and figures
```

## Installation

```bash
git clone https://github.com/yuvalira/Brain_Tumor_Segmentation.git
cd Brain_Tumor_Segmentation

python -m pip install "numpy==1.26.4" scipy pandas matplotlib \
    scikit-learn opencv-python h5py optuna jupyter
```

## Running the Project

Place the converted BraTS HDF5 dataset in the directory expected by `utilities/utils.py`.

If normalization statistics are missing, run:

```bash
python utilities/compute_z_score_stats.py
```

Then open and run the main notebook:

```bash
jupyter notebook main.ipynb
```

The notebook trains or loads the GMMs, selects hyperparameters using the validation set, evaluates the frozen models on the test set, and generates the required tables and figures.

## Authors

Etamar Rothstein and Yuval Ratzabi
Ben-Gurion University of the Negev

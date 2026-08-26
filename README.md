# Brain Tumor Segmentation from Multimodal MRI

A classical anatomy-aware brain-tumor segmentation framework developed for the **Magnetic Resonance Imaging (361.2.6501) Final Project** at Ben-Gurion University of the Negev.

The project compares a raw-intensity Gaussian Mixture Model (GMM) baseline with an advanced model that incorporates **hemispheric symmetry** and **distance from the brain boundary**. Both models use the same spatial image-processing pipeline to convert voxel-wise tumor probabilities into a binary whole-tumor segmentation.

## Project Overview

Each BraTS case contains four aligned MRI modalities:

* T1-weighted MRI
* Contrast-enhanced T1-weighted MRI (T1ce)
* T2-weighted MRI
* Fluid-Attenuated Inversion Recovery (FLAIR)

The provided tumor subregion annotations are combined into a single binary **whole-tumor mask**.

To reduce computational requirements, the experiment processes **axial slice 80** from every patient. The same slice-selection rule is applied to all models and data splits.

## Dataset

The project uses the **BraTS 2020 training dataset**, containing 369 patient volumes.

The patients are divided into mutually exclusive subsets:

| Split      | Patients | Purpose                     |
| ---------- | -------: | --------------------------- |
| Training   |      250 | GMM estimation              |
| Validation |       50 | Hyperparameter optimization |
| Test       |       69 | Final evaluation            |

The split is performed at the patient level with a fixed random seed. Stratification accounts for central-slice tumor size and patient-order groups. The exact split is saved and reused by every model.

The dataset is not included in this repository.

## Preprocessing

For every patient:

1. A foreground brain mask is extracted from the nonzero MRI voxels.
2. Each MRI modality is normalized independently using volume-level z-score normalization.
3. Background voxels outside the brain mask are excluded.
4. The four tumor subregion annotations are converted into one binary whole-tumor mask.

Volume-level normalization statistics are calculated using the complete 3D volume, although segmentation is performed only on axial slice 80.

## Baseline Model

The baseline represents each brain voxel using the four normalized MRI intensities:

$$
\mathbf{f}_{raw}(x)=
[T1(x),T1ce(x),T2(x),FLAIR(x)]^T
$$

Separate Gaussian Mixture Models are trained for:

* Healthy brain tissue
* Necrotic tumor core
* Peritumoral edema
* Enhancing tumor

The healthy-tissue GMM uses nine components. Each tumor-tissue GMM uses three components.

The class likelihoods and training-set class priors are combined using Bayes’ rule to calculate voxel-wise posterior tumor probabilities.

## Advanced Boundary + Symmetry Model

The advanced model augments the four MRI intensities with two forms of anatomical information.

### Hemispheric Symmetry

Four Normalized Difference Index channels describe the intensity difference between corresponding locations in the two cerebral hemispheres.

These features help detect unilateral pathological changes that disrupt the approximate bilateral symmetry of healthy brain tissue.

### Boundary Depth

A normalized Euclidean distance-transform channel represents the relative depth of each voxel from the outer brain boundary.

This feature provides spatial context and helps distinguish tumor tissue from healthy structures with similar MRI intensity profiles.

The final anatomy-aware representation contains nine features:

$$
\mathbf{f}_{advanced}(x)=
[\mathbf{f}_{raw}(x),\mathbf{f}_{symmetry}(x),d(x)]^T
\in\mathbb{R}^{9}
$$

## Spatial Image-Processing Pipeline

The GMM produces point-wise posterior probabilities rather than a spatially connected segmentation. The posterior maps are therefore refined using a shared classical image-processing pipeline:

1. Sobel edge detection on the merged tumor-posterior map
2. Scaled Otsu thresholding
3. Morphological closing
4. Closed-contour extraction
5. Minimum component-size filtering
6. Entropy-weighted contour classification
7. Constrained region expansion from accepted tumor seeds
8. Binary whole-tumor segmentation

The structural image-processing parameters are optimized for the baseline using the validation set and then frozen for the anatomy-aware models. Only probability-dependent thresholds are recalibrated separately for each feature representation.

## Hyperparameter Selection

Hyperparameters are selected exclusively using the validation set with Optuna and a fixed random seed.

The baseline optimization jointly selects:

* Minimum contour size
* Sobel binarization factor
* Internal-contour retrieval
* Maximum expansion distance
* Posterior contour-classification threshold
* Entropy expansion threshold
* Posterior expansion threshold

The improved models reuse the frozen structural parameters and optimize only their three probability-dependent thresholds.

No GMM fitting or hyperparameter selection is performed using the final test results.

## Results

Performance is evaluated using the Dice Similarity Coefficient and Intersection over Union for every test patient.

| Model                   |                Dice |                 IoU | Missed tumors | Empty-slice FP |
| ----------------------- | ------------------: | ------------------: | ------------: | -------------: |
| 4D Raw Baseline         |     0.6828 ± 0.3373 |     0.5998 ± 0.3231 |         10/62 |            2/7 |
| 5D Boundary Depth       |     0.7061 ± 0.3270 |     0.6234 ± 0.3123 |          9/62 |            2/7 |
| 8D Hemispheric Symmetry | **0.7323 ± 0.2769** | **0.6361 ± 0.2737** |          6/62 |        **1/7** |
| 9D Boundary + Symmetry  |     0.7215 ± 0.2732 |     0.6218 ± 0.2751 |      **4/62** |            2/7 |

Hemispheric symmetry achieves the highest mean Dice and IoU, while the combined Boundary + Symmetry representation produces the fewest complete tumor misses.

## Evaluation Outputs

The notebook produces:

* Mean and standard deviation of Dice and IoU
* Per-patient Dice and IoU distributions
* Dice and IoU boxplots
* Paired baseline-versus-model scatterplots
* Pearson correlation coefficients
* Missed-tumor and false-positive counts
* Qualitative segmentation comparisons
* Detailed error maps for completely missed tumors
* Intermediate posterior, entropy, contour, and expansion visualizations

## Repository Structure

```text
Brain_Tumor_Segmentation/
├── main.ipynb
├── config.py
├── evaluation/
│   ├── evaluate_single_slice.py
│   ├── evaluate_test_set.py
│   ├── hyperparameter_optimization.py
│   ├── boxplots_visualization.py
│   └── scatterplot_visualization.py
├── image_processing/
│   ├── compute_entropy.py
│   ├── edge_detection.py
│   ├── contour_detection.py
│   ├── contour_classification.py
│   ├── seed_expansion.py
│   └── visualizations.py
├── statistical_models/
│   ├── train_healthy.py
│   ├── train_tumor.py
│   ├── healthy_likelihood.py
│   └── tumor_likelihoods.py
├── utilities/
│   ├── patient_split.py
│   ├── compute_z_score_stats.py
│   └── utils.py
├── saved_parameters/
└── output/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yuvalira/Brain_Tumor_Segmentation.git
cd Brain_Tumor_Segmentation
```

Install the required Python packages:

```bash
python -m pip install "numpy==1.26.4" scipy pandas matplotlib \
    scikit-learn opencv-python h5py optuna jupyter
```

## Dataset Location

The code expects the converted BraTS HDF5 slices to follow this structure relative to the repository:

```text
MRI_2026_datasets/
└── Brats/
    └── BraTS2020_training_data/
        └── content/
            └── data/
                ├── volume_1_slice_0.h5
                ├── volume_1_slice_1.h5
                └── ...
```

Each HDF5 file must contain:

* `image`: four-channel MRI image
* `mask`: tumor subregion annotation

The dataset path can be adjusted in `utilities/utils.py` if necessary.

## Running the Project

If the volume-level normalization files are not already available, generate them first:

```bash
python utilities/compute_z_score_stats.py
```

Then launch Jupyter:

```bash
jupyter notebook main.ipynb
```

Run the notebook from top to bottom.

The notebook will:

1. Create or load the fixed patient split
2. Calculate and display split statistics
3. Train or load the baseline GMM
4. Optimize the baseline parameters on validation data
5. Train the anatomy-aware GMMs
6. Optimize their probability thresholds
7. Evaluate all frozen models on the test set
8. Generate the required tables, plots, and qualitative examples

The initial full run can take several hours because each Optuna trial evaluates all validation patients. Saved parameters are reused in subsequent runs.

## Reproducibility

Reproducibility is supported through:

* Patient-level data separation
* Fixed random seeds
* A saved and reusable patient split
* Training-volume identifiers stored with each fitted GMM
* Validation-only hyperparameter optimization
* Frozen parameters before final test evaluation
* Modular model, image-processing, and evaluation code

## Limitations

* Only one axial slice is processed per patient.
* Small tumors may be removed by the minimum component-size criterion.
* Bilateral symmetry may be affected by anatomical asymmetry, mass effect, or imperfect midline alignment.
* The GMM assumes that the training distributions generalize across patients.
* The spatial contour pipeline may reject large atypical tumors when their posterior confidence is insufficient.

## Conclusion

Adding anatomical information improves the intensity-only GMM baseline. Hemispheric symmetry provides the strongest improvement in average segmentation accuracy, while combining symmetry with boundary depth reduces the number of completely missed tumors.

The results demonstrate that lightweight anatomical priors can improve classical statistical segmentation without requiring atlas registration or deep-learning training.

## Authors

* Etamar Rothstein 
* Yuval Ratzabi

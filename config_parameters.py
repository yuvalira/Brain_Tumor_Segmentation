from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DATASET_PATH = (
    WORKSPACE_ROOT
    / "MRI_2026_datasets"
    / "Brats"
    / "BraTS2020_training_data"
    / "content"
    / "data"
)

UTILITIES_PATH = PROJECT_ROOT / "utilities"
OUTPUT_FIGURES_PATH = PROJECT_ROOT / "output_figures"

VOLUME_MEANS_PATH = UTILITIES_PATH / "volume_means.npy"
VOLUME_STDS_PATH = UTILITIES_PATH / "volume_stds.npy"


# =============================================================================
# DATASET
# =============================================================================

FIRST_VOLUME = 1
MAX_TRAINING_VOLUME = 300
FIRST_TEST_VOLUME = MAX_TRAINING_VOLUME + 1
TOTAL_VOLUMES = 369

NUM_SLICES = 155
MAX_SLICE = NUM_SLICES - 1

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240

MODALITY_NAMES = (
    "T1",
    "T1ce",
    "T2",
    "FLAIR",
)

NUM_MODALITIES = len(MODALITY_NAMES)


# =============================================================================
# SEMANTIC CLASSES AND POSTERIOR CHANNELS
# =============================================================================

HEALTHY_CLASS_NAME = "HEALTHY"

TUMOR_CLASS_NAMES = (
    "NCR_NET",
    "ED",
    "ET",
)

POSTERIOR_CLASS_NAMES = (
    HEALTHY_CLASS_NAME,
    *TUMOR_CLASS_NAMES,
)

NUM_TUMOR_CLASSES = len(TUMOR_CLASS_NAMES)
NUM_POSTERIOR_CLASSES = len(POSTERIOR_CLASS_NAMES)

# Posterior channel order:
#
# 0: Healthy
# 1: NCR/NET
# 2: ED
# 3: ET
#
HEALTHY_POSTERIOR_INDEX = 0
TUMOR_POSTERIOR_INDICES = (1, 2, 3)


# =============================================================================
# STATISTICAL MODEL
# =============================================================================

# Available options:
#     "gaussian"
#     "skew_t"
#     "gmm"       # Future implementation
STATISTICAL_MODEL = "skew_t"


# =============================================================================
# PREPROCESSING
# =============================================================================

BRAIN_MASK_THRESHOLD = 1e-8

ZSCORE_CLIP_MIN = -6.0
ZSCORE_CLIP_MAX = 6.0


# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================

RANDOM_SEED = 42

HEALTHY_SKEW_T_MAX_FIT_SAMPLES = 3_000_000
TUMOR_SKEW_T_MAX_FIT_SAMPLES = 100_000

# These parameters are intended for the future healthy GMM.
HEALTHY_GMM_COMPONENTS = 9

HEALTHY_TISSUE_NAMES = (
    "CSF",
    "GM",
    "WM",
)


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

PROBABILITY_EPSILON = 1e-12


# =============================================================================
# CONTOUR DETECTION
# =============================================================================

MIN_NUM_PIXELS_PER_BLOB_DEFAULT = 50
SOBEL_BINARIZATION_OTSU_FACTOR = 1.0


# =============================================================================
# CONTOUR CLASSIFICATION
# =============================================================================

WEIGHTED_POSTERIOR_MEAN_THRESHOLD = 0.5


# =============================================================================
# SEED EXPANSION
# =============================================================================

ENTROPY_THRESHOLD_DEFAULT = 0.25
POSTERIOR_THRESHOLD_DEFAULT = 0.05
MAX_EXPANSION_DIAMETER_DEFAULT = 20


# =============================================================================
# DEFAULT VISUALIZATION / DEBUG EXAMPLE
# =============================================================================

DEFAULT_VOLUME_NUM = 350
DEFAULT_SLICE_NUM = 90
DEFAULT_TARGET_ROW = 115
FIGURE_DPI = 300

# =============================================================================
# FINAL GMM COMPONENT COUNTS
# =============================================================================

HEALTHY_GMM_COMPONENTS = 9

TUMOR_GMM_COMPONENTS = {
    "NCR_NET": 4,
    "ED": 4,
    "ET": 4,
}
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent

MODELS_DIR = PROJECT_ROOT / "saved_parameters" / "statistical_models"

MAX_TRAINING_VOLUME = 250
MAX_VALIDATION_VOLUME = 300
TOTAL_VOLUMES = 369
NUMBER_OF_VOLUMES = 369
NUMBER_OF_SLICES = 155
NUM_HEALTHY_TRAINING_SAMPLES = 3_000_000
SLICE_NUM = 80

GMM_RAW_COMPONENTS   = 9
GMM_SYMMETRIC_COMPONENTS = 9
GMM_BOUNDARY_COMPONENTS  = 9
GMM_FULL_COMPONENTS      = 9
GMM_TUMOR_COMPONENTS     = 3


RANDOM_SEED = 42
IMAGE_PIXEL_LENGTH = 240

#-----------------------------------------------
#         Image Processing Parameters
#-----------------------------------------------

# contour detection
MIN_NUM_PIXELS_PER_BLOB_DEFAULT = 22
SOBEL_BINARIZATION_OTSU_FACTOR  = 0.58
ALLOW_INTERNAL_CONTOURS = True

# seed expansion
MAX_EXPANSION_DIAMETER_DEFAULT = 100


#-----------------------------------------------
#         Statistical Thresholds
#-----------------------------------------------

# Model: RAW (Validation Mean Dice: 0.7531)
WEIGHTED_POSTERIOR_MEAN_THRESHOLD_RAW = 0.60
ENTROPY_THRESHOLD_RAW                 = 0.18
POSTERIOR_THRESHOLD_RAW               = 0.20

# Model: SYMMETRIC (Validation Mean Dice: 0.7643)
WEIGHTED_POSTERIOR_MEAN_THRESHOLD_SYMMETRIC = 0.53
ENTROPY_THRESHOLD_SYMMETRIC                 = 0.08
POSTERIOR_THRESHOLD_SYMMETRIC               = 0.17

# Model: BOUNDARY_DISTANCE (Validation Mean Dice: 0.7784)
WEIGHTED_POSTERIOR_MEAN_THRESHOLD_BOUNDARY_DISTANCE = 0.63
ENTROPY_THRESHOLD_BOUNDARY_DISTANCE                 = 0.16
POSTERIOR_THRESHOLD_BOUNDARY_DISTANCE               = 0.05

# Model: ALL (Validation Mean Dice: 0.7540)
WEIGHTED_POSTERIOR_MEAN_THRESHOLD_ALL = 0.64
ENTROPY_THRESHOLD_ALL                 = 0.10
POSTERIOR_THRESHOLD_ALL               = 0.42

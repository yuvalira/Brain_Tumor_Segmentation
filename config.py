import os
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MAX_TRAINING_VOLUME = 250
MAX_VALIDATION_VOLUME = 300
TOTAL_VOLUMES = 369
NUM_HEALTHY_TRAINING_SAMPLES = 3_000_000
SLICE_NUM = 80
GMM_SYMMETRIC_COMPONENTS = 24
GMM_REGULAR_COMPONENTS = 12
RANDOM_SEED = 42
IMAGE_PIXEL_LENGTH = 240
MODEL = "regular" # "regular" / "symmetric"

#-----------------------------------------------
#         Image Processing Parameters
#-----------------------------------------------
LAMBDA = 0.23

# contour detection
MIN_NUM_PIXELS_PER_BLOB_DEFAULT = 22
SOBEL_BINARIZATION_OTSU_FACTOR  = 0.58
ALLOW_INTERNAL_CONTOURS = True

# contour classification
WEIGHTED_POSTERIOR_MEAN_THRESHOLD = 0.33

# seed expansion
ENTROPY_THRESHOLD_DEFAULT      = 0.1
POSTERIOR_THRESHOLD_DEFAULT    = 0.33
MAX_EXPANSION_DIAMETER_DEFAULT = 24
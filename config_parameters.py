from pathlib import Path

DATASET_PATH                = "MRI_2026_datasets/Brats/BraTS2020_training_data/content/data"
PARAMS_OUTPUT_PATH          = "computed_parameters"
MAX_TRAINING_VOLUME         = 300
TOTAL_VOLUMES               = 369
MAX_SLICE                   = 154
HEALTHY_GAUSSIAN_COMPONENTS = 9
HEALTHY_CLASSES             = ['CSF', 'GM', 'WM']
ROOT_PATH = Path(__file__).resolve().parent

#-----------------------------------------------
#            Statistical Model
#-----------------------------------------------

MODEL = "gaussian" # "gaussian" / "skew_t" / "gmm"

#-----------------------------------------------
#         Image Processing Parameters
#-----------------------------------------------

# contour detection
MIN_NUM_PIXELS_PER_BLOB_DEFAULT = 50
SOBEL_BINARIZATION_OTSU_FACTOR  = 1

# contour classification
WEIGHTED_POSTERIOR_MEAN_THRESHOLD = 0.1

# seed expansion
ENTROPY_THRESHOLD_DEFAULT      = 0.25
POSTERIOR_THRESHOLD_DEFAULT    = 0.05
MAX_EXPANSION_DIAMETER_DEFAULT = 20
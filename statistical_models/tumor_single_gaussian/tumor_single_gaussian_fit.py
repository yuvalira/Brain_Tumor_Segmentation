import os
import numpy as np
from pathlib import Path
import sys


# Add project root ('Brain_Tumor_Segmentation') to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Now your import will resolve cleanly!
from utilities.utils import load_and_normalize_slice
from config_parameters import *



def estimate_tumor_parameters(
    max_train_vol=MAX_TRAINING_VOLUME,
    total_slices=MAX_SLICE + 1,
    output_path=os.path.join(
        PARAMS_OUTPUT_PATH,
    ),
):
    """
    Estimates the multivariate Gaussian parameters for the
    three tumor classes:

        NCR/NET
        ED
        ET

    using maximum likelihood estimation.
    """

    print("\nEstimating tumor Gaussian parameters...")

    num_classes = 3
    num_dims = 4

    total_brain_voxels = np.uint64(0)

    running_count = np.zeros(
        num_classes,
        dtype=np.uint64,
    )

    running_sum = np.zeros(
        (num_classes, num_dims),
        dtype=np.float64,
    )

    running_sum_squares = np.zeros(
        (num_classes, num_dims, num_dims),
        dtype=np.float64,
    )

    for vol_num in range(1, max_train_vol + 1):
        print(f'volume: {vol_num}/{MAX_TRAINING_VOLUME}')
        for slice_num in range(total_slices):

            (
                norm_slice,
                brain_mask,
                mask_slice,
            ) = load_and_normalize_slice(
                vol_num,
                slice_num,
            )

            if not np.any(brain_mask):
                continue

            image_flat = norm_slice.reshape(-1, num_dims)
            mask_flat = mask_slice.reshape(-1, 3)
            brain_flat = brain_mask.ravel()

            total_brain_voxels += np.count_nonzero(brain_flat)

            class_masks = [
                (mask_flat[:, 0] > 0) & brain_flat,
                (mask_flat[:, 1] > 0) & brain_flat,
                (mask_flat[:, 2] > 0) & brain_flat,
            ]

            for class_idx, class_mask in enumerate(class_masks):

                if not np.any(class_mask):
                    continue

                pixels = image_flat[class_mask]

                running_count[class_idx] += pixels.shape[0]

                running_sum[class_idx] += np.sum(
                    pixels,
                    axis=0,
                )

                running_sum_squares[class_idx] += (
                    pixels.T @ pixels
                )

        if (
            vol_num % 50 == 0
            or vol_num == max_train_vol
        ):
            print(
                f"Processed "
                f"{vol_num}/{max_train_vol}"
            )


    tumor_priors = (
            running_count.astype(np.float64)
            / float(total_brain_voxels)
    )

    tumor_means = np.zeros(
        (num_classes, num_dims),
        dtype=np.float64,
    )

    tumor_covariances = np.zeros(
        (num_classes, num_dims, num_dims),
        dtype=np.float64,
    )

    for class_idx in range(num_classes):

        tumor_means[class_idx] = (
            running_sum[class_idx]
            / running_count[class_idx]
        )

        mu = tumor_means[class_idx]

        tumor_covariances[class_idx] = (
            running_sum_squares[class_idx]
            / running_count[class_idx]
            - np.outer(mu, mu)
        )


    np.savez(
        f'Brain_Tumor_Segmentation/statistical_models/tumor_single_gaussian/tumor_single_gaussian_parameters.npz',

        priors=tumor_priors,

        means=tumor_means,

        covariances=tumor_covariances,

        counts=running_count,

        class_names=np.array(
            [
                "NCR_NET",
                "ED",
                "ET",
            ]
        ),

        modalities=np.array(
            [
                "T1",
                "T1ce",
                "T2",
                "FLAIR",
            ]
        ),
    )

    print("\nTumor Gaussian parameters saved to")
    print(output_path)

    print("\nTumor priors")
    print("-----------------------")

    for i, name in enumerate(
        [
            "NCR/NET",
            "ED",
            "ET",
        ]
    ):
        print(
            f"{name:8s}: "
            f"{tumor_priors[i]:.6f}"
        )

    print(
        "Sum      : "
        f"{tumor_priors.sum():.6f}"
    )

    return (
        tumor_priors,
        tumor_means,
        tumor_covariances,
    )

if __name__ == "__main__":


    tumor_priors, tumor_means, tumor_covariances = (
        estimate_tumor_parameters(
            max_train_vol=MAX_TRAINING_VOLUME,
            total_slices=MAX_SLICE + 1,
        )
    )

    print("\nFinished estimating tumor parameters.")
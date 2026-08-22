import os
from config import *
import numpy as np
from sklearn.mixture import GaussianMixture
from utilities.utils import load_and_normalize_slice


def fit_and_save_healthy_gmm(num_components, filename, channel_indices):
  """Fits a Gaussian Mixture Model on healthy brain tissue voxels across an arbitrary

  subset of feature channels and saves the learned parameters.

  :param num_components: Number of mixture components (K).
  :param filename: Output filename (e.g., 'healthy_gmm_4d.npz',
  'healthy_gmm_9d.npz').
  :param channel_indices: List or range of channel indices to select from the 9D
  tensor.
  """
  output_dir = os.path.join(
      PROJECT_ROOT, "saved_parameters", "statistical_models"
  )
  os.makedirs(output_dir, exist_ok=True)
  output_file = os.path.join(output_dir, filename)

  num_features = len(channel_indices)

  # Pre-allocate buffer for sampling: shape (num_features, N)
  healthy_pixels = np.zeros(
      (num_features, NUM_HEALTHY_TRAINING_SAMPLES), dtype=np.float64
  )
  sampled_pixel_counter = 0

  total_dataset_healthy_pixels = 0
  total_dataset_brain_pixels = 0

  print(
      f"Processing training volumes 1 to {MAX_TRAINING_VOLUME} | Channels:"
      f" {channel_indices} | K={num_components}..."
  )

  for vol_num in range(1, MAX_TRAINING_VOLUME + 1):
    image, brain_mask, mask, _ = load_and_normalize_slice(vol_num, SLICE_NUM)

    # Slice only the requested feature channels
    features_image = image[:, :, channel_indices]

    # Ground truth tumor mask
    binary_gt_mask = np.any(mask > 0, axis=-1) if mask.ndim == 3 else (mask > 0)

    # Isolated healthy brain tissue mask
    healthy_mask = brain_mask & (~binary_gt_mask)

    n_brain = np.sum(brain_mask)
    n_healthy = np.sum(healthy_mask)

    total_dataset_brain_pixels += n_brain
    total_dataset_healthy_pixels += n_healthy

    # Sample healthy voxels into buffer
    if (sampled_pixel_counter < NUM_HEALTHY_TRAINING_SAMPLES) and (
        n_healthy > 0
    ):
      valid_features = features_image[healthy_mask]  # Shape: (N_valid, D)
      num_valid = valid_features.shape[0]
      remaining_capacity = NUM_HEALTHY_TRAINING_SAMPLES - sampled_pixel_counter

      if num_valid > remaining_capacity:
        valid_features = valid_features[:remaining_capacity]
        num_valid = remaining_capacity

      # Transpose into shape (D, num_valid) to fit buffer columns
      healthy_pixels[
          :, sampled_pixel_counter : sampled_pixel_counter + num_valid
      ] = valid_features.T
      sampled_pixel_counter += num_valid

      if vol_num % 10 == 0 or sampled_pixel_counter >= NUM_HEALTHY_TRAINING_SAMPLES:
        print(
            f"Vol {vol_num}/{MAX_TRAINING_VOLUME} | Feature Buffer:"
            f" {sampled_pixel_counter}/{NUM_HEALTHY_TRAINING_SAMPLES}"
        )

  # Trim to actual sampled count
  healthy_pixels = healthy_pixels[:, :sampled_pixel_counter]

  # Empirical prior P(Healthy)
  healthy_prior = (
      total_dataset_healthy_pixels / total_dataset_brain_pixels
      if total_dataset_brain_pixels > 0
      else 1.0
  )

  print("\n--- Model Training Summary ---")
  print(f"Total Dataset Brain Pixels: {total_dataset_brain_pixels}")
  print(f"Total Dataset Healthy Pixels: {total_dataset_healthy_pixels}")
  print(f"Dataset Prior P(Healthy): {healthy_prior:.6f}")
  print(f"Training Matrix Shape: {healthy_pixels.T.shape}")

  # Fit GMM: shape (N_samples, D)
  X_train = healthy_pixels.T

  gmm = GaussianMixture(
      n_components=num_components,
      covariance_type="full",
      max_iter=200,
      reg_covar=1e-5,
      init_params="random_from_data",
      random_state=RANDOM_SEED,
      n_init=5,
  )
  gmm.fit(X_train)

  np.savez(
      output_file,
      prior=healthy_prior,
      weights=gmm.weights_,
      means=gmm.means_,
      covariances=gmm.covariances_,
      channel_indices=np.array(channel_indices),
  )
  print(f"GMM parameters successfully saved to '{output_file}'\n")

if __name__ == "__main__":
    # Example function calls
    fit_and_save_healthy_gmm()
    fit_and_save_healthy_gmm()
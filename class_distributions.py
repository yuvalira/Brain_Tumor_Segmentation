import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t
from scipy.special import stdtr  # Student's t CDF for skewing
from scipy.optimize import curve_fit

# Import custom utility function
from utils import load_volume_stats

# --- Parameters & Paths ---
DATASET_PATH = "MRI_2026_datasets/Brats/BraTS2020_training_data/content/data"
STATS_DIR = "data"
OUTPUT_DIR = "tumor_class_histograms_skewt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOLUMES_RANGE = range(1, 241)  # Volumes 1 to 240
SLICES_PER_VOLUME = 155        # Slices 0 to 154
MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]

TARGET_CHANNELS = {
    "NCR_NET": 0,
    "Edema": 1,
    "ET": 2
}

# Fixed Range [-3.5, 3.5] and 200 Bins
BIN_MIN, BIN_MAX = -3.5, 3.5
NUM_BINS = 200
bin_edges = np.linspace(BIN_MIN, BIN_MAX, NUM_BINS + 1)
bin_width = bin_edges[1] - bin_edges[0]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

# Data Structures for Fast Online Accumulation
bin_counts = {
    c_name: np.zeros((4, NUM_BINS), dtype=np.int64) for c_name in TARGET_CHANNELS
}
total_voxels = {
    c_name: np.zeros(4, dtype=np.int64) for c_name in TARGET_CHANNELS
}
out_of_bounds_voxels = {
    c_name: np.zeros(4, dtype=np.int64) for c_name in TARGET_CHANNELS
}

# Load Saved Pre-computed Means and Stds
volume_means, volume_stds = load_volume_stats(STATS_DIR)

print("Processing slices and performing online binning...")

# Fast Online Accumulation Loop
for vol_idx in VOLUMES_RANGE:
    print(f'volume: {vol_idx}/240')
    vol_mean = volume_means[vol_idx]
    vol_std = volume_stds[vol_idx]
    vol_std = np.where(vol_std == 0, 1e-8, vol_std)

    for slice_num in range(SLICES_PER_VOLUME):
        h5_filename = f"volume_{vol_idx}_slice_{slice_num}.h5"
        h5_path = os.path.join(DATASET_PATH, h5_filename)
        
        if not os.path.exists(h5_path):
            continue

        with h5py.File(h5_path, 'r') as hf:
            image_data = hf['image'][:]  # (240, 240, 4)
            mask_data = hf['mask'][:]    # (240, 240, 3)

        norm_slice = (image_data - vol_mean) / vol_std

        for c_name, target_ch in TARGET_CHANNELS.items():
            class_mask = (mask_data[..., target_ch] > 0)
            
            if not np.any(class_mask):
                continue

            for mod_idx in range(4):
                voxels = norm_slice[..., mod_idx][class_mask]
                
                n_vox = len(voxels)
                total_voxels[c_name][mod_idx] += n_vox

                oob_mask = (voxels < BIN_MIN) | (voxels > BIN_MAX)
                out_of_bounds_voxels[c_name][mod_idx] += np.sum(oob_mask)

                in_bounds_voxels = voxels[~oob_mask]
                if len(in_bounds_voxels) > 0:
                    counts, _ = np.histogram(in_bounds_voxels, bins=bin_edges)
                    bin_counts[c_name][mod_idx] += counts

print("\nAccumulation complete! Fitting 4th-order Skew-t distributions...\n")

# --- Mathematical Fitting Functions ---

def fit_gaussian_pdf(x, y_density):
    """Fits Gaussian N(mu, sigma) - 1st & 2nd Order Statistics."""
    def g_func(x_val, loc, scale):
        return norm.pdf(x_val, loc=loc, scale=scale)
    try:
        popt, _ = curve_fit(g_func, x, y_density, p0=[0.0, 1.0], maxfev=5000)
        return g_func(x, *popt)
    except Exception:
        return norm.pdf(x, loc=0, scale=1)

def skew_t_pdf_func(x, a, df, loc, scale):
    """
    Azzalini-type Skewed Student's t PDF:
    f(x) = (2/scale) * t_pdf(z; df) * t_cdf(a * z * sqrt((df+1)/(df + z^2)); df+1)
    where z = (x - loc) / scale.
    Parameters:
      a    : 3rd-order Skewness
      df   : 4th-order Kurtosis (Degrees of Freedom)
      loc  : 1st-order Location
      scale: 2nd-order Dispersion
    """
    z = (x - loc) / scale
    pdf_t = student_t.pdf(z, df=df)
    
    # Argument transformation for the cumulative student-t factor
    w = a * z * np.sqrt((df + 1.0) / (df + z**2))
    cdf_t = stdtr(df + 1.0, w)
    
    return (2.0 / scale) * pdf_t * cdf_t

def fit_skew_t_pdf(x, y_density):
    """Fits 4th-order Skewed Student's t (a, df, loc, scale) using NLS."""
    # Initial guesses: skewness=1.0, df=4.0 (heavy tails), loc=mode(x), scale=std(x)
    p0 = [1.0, 4.0, 0.0, 1.0]
    # Parameter bounds: a in (-inf, inf), df in (1.001, 100), loc in (-3.5, 3.5), scale > 0
    bounds = ([-np.inf, 1.001, -3.5, 0.01], [np.inf, 100.0, 3.5, 5.0])
    
    try:
        popt, _ = curve_fit(skew_t_pdf_func, x, y_density, p0=p0, bounds=bounds, maxfev=10000)
        fitted_pdf = skew_t_pdf_func(x, *popt)
        return fitted_pdf, popt
    except Exception as e:
        # Fallback if convergence fails
        fallback_pdf = fit_gaussian_pdf(x, y_density)
        return fallback_pdf, [0.0, 100.0, 0.0, 1.0]

# --- Plotting Loop ---

for c_name in TARGET_CHANNELS:
    for mod_idx, mod_name in enumerate(MODALITIES):
        tot = total_voxels[c_name][mod_idx]
        oob = out_of_bounds_voxels[c_name][mod_idx]
        in_b = tot - oob
        oob_pct = (oob / tot * 100) if tot > 0 else 0.0

        if tot == 0 or in_b == 0:
            continue

        counts = bin_counts[c_name][mod_idx]
        density = counts / (in_b * bin_width)

        # Compute Fits
        gaussian_fit = fit_gaussian_pdf(bin_centers, density)
        skewt_fit, (a_fit, df_fit, loc_fit, scale_fit) = fit_skew_t_pdf(bin_centers, density)

        plt.figure(figsize=(9.5, 6))
        
        # 1. Empirical Density Histogram
        plt.bar(
            bin_centers, 
            density, 
            width=bin_width, 
            align='center', 
            alpha=0.45, 
            color='steelblue', 
            edgecolor='none',
            label=f'Empirical Density (Bins={NUM_BINS})'
        )
        
        # 2. Gaussian Fit (2nd Order)
        plt.plot(
            bin_centers, 
            gaussian_fit, 
            color='darkorange', 
            linestyle='-', 
            linewidth=2.2, 
            label='Gaussian Fit (Symmetric, $\\nu=\\infty$)'
        )
        
        # 3. Skewed Student's t Fit (3rd & 4th Order)
        plt.plot(
            bin_centers, 
            skewt_fit, 
            color='crimson', 
            linestyle='-', 
            linewidth=2.5, 
            label=f'Skew-$t$ Fit \nSkew $\\alpha={a_fit:.2f}$ \nKurtosis $\\nu={df_fit:.1f}$)'
        )

        # Formatting
        plt.xlim(BIN_MIN, BIN_MAX)
        plt.title(
            f"Parametric 4th-Order Density Fits vs Empirical Distribution\n"
            f"Class: {c_name} | Modality: {mod_name} (OOB: {oob_pct:.2f}%)", 
            fontsize=12, 
            fontweight='bold'
        )
        plt.xlabel("Normalized Intensity (Z-Score)", fontsize=10)
        plt.ylabel("Probability Density", fontsize=10)
        plt.grid(True, linestyle="-", alpha=0.5)
        
        # Legend with dynamic parameters
        plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

        # Save Plot
        save_filename = f"skewt_fitted_hist_{c_name}_{mod_name}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, save_filename), dpi=300, bbox_inches="tight")
        plt.close()

print(f"All Skew-t fitted histograms saved successfully to '{OUTPUT_DIR}'!")
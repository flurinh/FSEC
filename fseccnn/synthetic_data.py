# synthetic_data.py
# ... (imports and other functions like gaussian_peak_scipy, emg_peak_scipy, generate_smoothed_noise) ...
import numpy as np
from scipy.special import erfc
from scipy.stats import exponnorm, norm
from scipy.signal import savgol_filter, convolve
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import json  # If you decide to save peak_params as JSON strings


# --- Peak Shape Functions (ensure these are defined or imported correctly) ---
def gaussian_peak_scipy(x, amplitude, mean, std_dev):
    if std_dev <= 1e-6: std_dev = 1e-6
    return amplitude * norm.pdf(x, loc=mean, scale=std_dev)


def emg_peak_scipy(x, amplitude, mu, sigma, tau):
    if sigma <= 1e-6: sigma = 1e-6
    if tau <= 1e-6: tau = 1e-6
    K = tau / sigma
    return amplitude * exponnorm.pdf(x, K=K, loc=mu - tau, scale=sigma)


# --- Smoothed Noise Generation (ensure this is defined or imported correctly) ---
def generate_smoothed_noise(length, noise_level_amplitude, smoothing_window_size=5, smoothing_polyorder=2):
    if noise_level_amplitude <= 0: return np.zeros(length)
    white_noise = np.random.normal(0, noise_level_amplitude, length)
    if smoothing_window_size > 1 and smoothing_window_size < length and smoothing_window_size % 2 == 1:
        try:
            smoothed_noise = savgol_filter(white_noise, window_length=smoothing_window_size,
                                           polyorder=min(smoothing_polyorder, smoothing_window_size - 1))
        except ValueError:
            smoothed_noise = savgol_filter(white_noise, window_length=smoothing_window_size,
                                           polyorder=max(1, smoothing_window_size - 2))
    else:
        smoothed_noise = white_noise
    return smoothed_noise


# --- Synthetic Chromatogram Generation (with Y-Corrected BASELINE) ---
def generate_synthetic_chromatogram_mixed_smooth_noise(
        length=512, x_range=(0, 100), max_peaks=5, min_peaks=1,
        amplitude_range=(0.5, 2.0), std_dev_range=(2.0, 6.0),
        tau_range=(0.0, 1.5), peak_type_ratio=0.5,
        shoulder_peak_probability=0.3, shoulder_amplitude_ratio_range=(0.05, 0.3),
        shoulder_std_dev_ratio_range=(0.5, 1.0), shoulder_mu_offset_ratio_range=(-1.5, 1.5),
        shoulder_tau_factor_range=(0.0, 1.0),
        base_noise_level=0.01, noise_smoothing_window_size=7, noise_smoothing_polyorder=2,

        # Baseline Parameters for generating variations
        baseline_linear_drift_variation=0.1,  # Max total change due to linear drift (factor of typical_amplitude)
        baseline_wobble_amplitude_factor=0.05,  # Factor of typical_amplitude for wobble
        baseline_wobble_periods_range=(0.5, 2.0),
        # Y-correction: baseline will be shifted so its min is this value (can be slightly negative to allow signal to dip a bit)
        baseline_y_corrected_min_level=0.0  # Or a small negative like -0.01 * typical_amplitude
):
    x_coords = np.linspace(x_range[0], x_range[1], length, dtype=np.float32)
    chromatogram_true = np.zeros(length, dtype=np.float32)  # Signal without noise/baseline
    peak_maxima_indices_for_label = []
    peak_params_list = []
    typical_amplitude = np.mean(amplitude_range)
    num_main_peaks = np.random.randint(min_peaks, max_peaks + 1)
    generated_peak_mus = []

    # --- Peak Generation (same as before) ---
    for i in range(num_main_peaks):
        main_A = np.random.uniform(amplitude_range[0], amplitude_range[1])
        main_sigma = np.random.uniform(std_dev_range[0], std_dev_range[1])
        min_mu_spacing = main_sigma * 1.0
        min_allowed_mu = (max(generated_peak_mus) + min_mu_spacing) if generated_peak_mus else (
                    x_range[0] + main_sigma * 1.5)  # Ensure space from edge
        max_allowed_mu = x_range[1] - main_sigma * 1.5  # Ensure space from edge
        if min_allowed_mu >= max_allowed_mu or min_allowed_mu < x_range[0] or max_allowed_mu > x_range[1]:
            main_mu = np.random.uniform(x_range[0] + main_sigma * 1.5, x_range[1] - main_sigma * 1.5)
            if main_mu < x_range[0] or main_mu > x_range[1]: main_mu = np.mean(x_range)  # Fallback
        else:
            main_mu = np.random.uniform(min_allowed_mu, max_allowed_mu)
        generated_peak_mus.append(main_mu)
        main_params = {'is_shoulder': False, 'A': main_A, 'mu': main_mu, 'sigma': main_sigma}
        if np.random.rand() < peak_type_ratio:
            main_tau = np.random.uniform(tau_range[0], tau_range[1] * main_sigma)
            if main_tau < 1e-4 * main_sigma: main_tau = 0.0
            peak_component = emg_peak_scipy(x_coords, main_A, main_mu, main_sigma, main_tau)
            main_params.update({'type': 'emg', 'tau': main_tau})
        else:
            peak_component = gaussian_peak_scipy(x_coords, main_A, main_mu, main_sigma)
            main_params.update({'type': 'gaussian', 'tau': 0.0})
        chromatogram_true += peak_component;
        peak_params_list.append(main_params)
        if main_A > typical_amplitude * 0.1: peak_maxima_indices_for_label.append(np.argmax(peak_component))

        if np.random.rand() < shoulder_peak_probability:
            shoulder_A = main_A * np.random.uniform(shoulder_amplitude_ratio_range[0],
                                                    shoulder_amplitude_ratio_range[1])
            shoulder_sigma = main_sigma * np.random.uniform(shoulder_std_dev_ratio_range[0],
                                                            shoulder_std_dev_ratio_range[1])
            mu_offset = main_sigma * np.random.uniform(shoulder_mu_offset_ratio_range[0],
                                                       shoulder_mu_offset_ratio_range[1])
            shoulder_mu = np.clip(main_mu + mu_offset, x_range[0] + shoulder_sigma * 1.1,
                                  x_range[1] - shoulder_sigma * 1.1)  # Ensure within bounds a bit more
            shoulder_params = {'is_shoulder': True, 'A': shoulder_A, 'mu': shoulder_mu, 'sigma': shoulder_sigma}
            if np.random.rand() < peak_type_ratio:
                shoulder_tau = np.random.uniform(shoulder_tau_factor_range[0],
                                                 shoulder_tau_factor_range[1]) * shoulder_sigma
                if shoulder_tau < 1e-4 * shoulder_sigma: shoulder_tau = 0.0
                shoulder_component = emg_peak_scipy(x_coords, shoulder_A, shoulder_mu, shoulder_sigma, shoulder_tau)
                shoulder_params.update({'type': 'emg', 'tau': shoulder_tau})
            else:
                shoulder_component = gaussian_peak_scipy(x_coords, shoulder_A, shoulder_mu, shoulder_sigma)
                shoulder_params.update({'type': 'gaussian', 'tau': 0.0})
            chromatogram_true += shoulder_component;
            peak_params_list.append(shoulder_params)
            if shoulder_A > typical_amplitude * 0.08:
                max_idx_shoulder = np.argmax(shoulder_component)
                is_far_enough = all(
                    abs(max_idx_shoulder - ex_idx) >= (shoulder_sigma / (x_range[1] - x_range[0] + 1e-6) * length * 0.3)
                    for ex_idx in peak_maxima_indices_for_label)
                if is_far_enough: peak_maxima_indices_for_label.append(max_idx_shoulder)
    # --- END Peak Generation ---

    # --- BASELINE GENERATION with Y-Correction ---
    # 1. Generate Linear Drift component (can start anywhere, will be shifted)
    x_range_val_safe = (x_range[1] - x_range[0]) if (x_range[1] - x_range[0]) > 0 else 1.0
    total_drift = np.random.uniform(-baseline_linear_drift_variation,
                                    baseline_linear_drift_variation) * typical_amplitude
    linear_drift_component = np.linspace(0, total_drift, length)  # Starts at 0, ends at total_drift

    # 2. Generate Wobble component (centered around zero initially)
    wobble_amp_val = baseline_wobble_amplitude_factor * typical_amplitude
    num_wobble_periods = np.random.uniform(baseline_wobble_periods_range[0], baseline_wobble_periods_range[1])
    wobble_phase = np.random.rand() * 2 * np.pi
    wobble_component = wobble_amp_val * np.sin(
        2 * np.pi * num_wobble_periods * (x_coords - x_range[0]) / x_range_val_safe + wobble_phase
    )

    # 3. Combine raw baseline components
    raw_baseline = linear_drift_component + wobble_component

    # 4. Y-Correction: Shift the raw baseline so its minimum is at baseline_y_corrected_min_level
    min_raw_baseline = np.min(raw_baseline)
    # Determine the target minimum for the baseline. Can be slightly negative.
    # If baseline_y_corrected_min_level is absolute:
    # target_min_baseline_val = baseline_y_corrected_min_level
    # If baseline_y_corrected_min_level is a factor of typical_amplitude (e.g. -0.01 for -1% of typical amp):
    target_min_baseline_val = baseline_y_corrected_min_level * typical_amplitude

    y_corrected_baseline = raw_baseline - min_raw_baseline + target_min_baseline_val

    total_baseline = y_corrected_baseline
    # --- END Baseline Generation ---

    chromatogram_with_baseline = chromatogram_true + total_baseline

    # --- Noise Generation ---
    noise_amp = base_noise_level * typical_amplitude
    smoothed_noise = generate_smoothed_noise(length, noise_amp, noise_smoothing_window_size, noise_smoothing_polyorder)
    final_chromatogram = chromatogram_with_baseline + smoothed_noise
    final_chromatogram = np.clip(final_chromatogram, 0, None)

    # --- Create Target Map for CNN ---
    target_map = np.zeros(length, dtype=np.float32)
    unique_maxima_indices_for_label = sorted(list(set(peak_maxima_indices_for_label)))
    target_peak_width_pixels = 5
    for idx in unique_maxima_indices_for_label:
        start = max(0, idx - target_peak_width_pixels // 2)
        end = min(length, idx + target_peak_width_pixels // 2 + 1)
        target_map[start:end] = 1.0

    return final_chromatogram.astype(np.float32), target_map.astype(np.float32), peak_params_list, x_coords


# --- PyTorch Dataset Class (LargeSyntheticFSECDataset - unchanged) ---
class LargeSyntheticFSECDataset(Dataset):
    def __init__(self, hdf5_path, epoch_subset_size=None, transform=None, target_transform=None):
        self.hdf5_path = hdf5_path;
        self.epoch_subset_size = epoch_subset_size
        self.transform = transform;
        self.target_transform = target_transform
        try:  # Add try-except for file access
            with h5py.File(self.hdf5_path, 'r') as db:
                self.total_samples = db['chromatograms'].shape[0]
                self.signal_length = db['chromatograms'].shape[1]
        except Exception as e:
            print(f"Error opening or reading HDF5 file {self.hdf5_path}: {e}")
            raise  # Re-raise the exception to stop execution if file is crucial

        self.current_indices = np.arange(self.total_samples)
        if self.epoch_subset_size is not None and self.epoch_subset_size < self.total_samples:
            if self.epoch_subset_size <= 0:  # Ensure subset_size is positive
                print(f"Warning: epoch_subset_size ({self.epoch_subset_size}) is not positive. Using all samples.")
                self.epoch_subset_size = None
            else:
                self.reshuffle_indices()
        elif self.epoch_subset_size is not None and self.epoch_subset_size >= self.total_samples:
            self.epoch_subset_size = None  # Use all samples if subset is larger or equal

        print(
            f"Dataset from {hdf5_path}. Total: {self.total_samples}. Subset per epoch: {self.epoch_subset_size if self.epoch_subset_size else 'All'}.")

    def reshuffle_indices(self):
        if self.epoch_subset_size is not None and self.epoch_subset_size < self.total_samples:
            self.current_indices = np.random.choice(self.total_samples, self.epoch_subset_size, replace=False)
        else:  # No subsetting, or subset is all samples
            self.current_indices = np.arange(self.total_samples)

    def __len__(self):
        return self.epoch_subset_size if (
                    self.epoch_subset_size is not None and self.epoch_subset_size < self.total_samples) else self.total_samples

    def __getitem__(self, idx):
        if idx >= len(self.current_indices):  # Should not happen if __len__ is correct
            raise IndexError(f"Index {idx} out of bounds for current_indices with length {len(self.current_indices)}")
        actual_idx = self.current_indices[idx]

        try:  # Add try-except for file access during getitem
            with h5py.File(self.hdf5_path, 'r') as db:
                chromatogram = db['chromatograms'][actual_idx, :].astype(np.float32)
                target_map = db['target_maps'][actual_idx, :].astype(np.float32)
        except Exception as e:
            print(f"Error reading sample index {actual_idx} (mapped from {idx}) from HDF5 file {self.hdf5_path}: {e}")
            # Return a dummy sample or raise error, depending on desired robustness
            # For now, let's raise it to be aware of issues
            raise

        sample_tensor = torch.from_numpy(chromatogram).unsqueeze(0)
        label_tensor = torch.from_numpy(target_map).unsqueeze(0)
        if self.transform: sample_tensor = self.transform(sample_tensor)
        if self.target_transform: label_tensor = self.target_transform(label_tensor)
        return sample_tensor, label_tensor


# --- Function to Generate and Save Large Dataset (create_large_offline_dataset - unchanged) ---
def create_large_offline_dataset(filepath, num_total_samples, generator_fn, generator_config, chunk_size=1000):
    if os.path.exists(filepath):
        overwrite = input(f"File {filepath} exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Skipping generation."); return
        else:
            os.remove(filepath)  # Remove before creating anew

    signal_length = generator_config.get('length', 256)
    with h5py.File(filepath, 'w') as hf:
        # Use chunks that align with typical access patterns, e.g., one sample
        dset_chroms = hf.create_dataset('chromatograms', shape=(num_total_samples, signal_length),
                                        maxshape=(None, signal_length), dtype='float32', chunks=(1, signal_length))
        dset_maps = hf.create_dataset('target_maps', shape=(num_total_samples, signal_length),
                                      maxshape=(None, signal_length), dtype='float32', chunks=(1, signal_length))
        dt_peak_params = h5py.string_dtype(encoding='utf-8')  # For JSON strings
        dset_peak_params = hf.create_dataset('peak_params', shape=(num_total_samples,), maxshape=(None,),
                                             dtype=dt_peak_params, chunks=(chunk_size,))

        print(f"Generating {num_total_samples} samples for {filepath} (chunk size {chunk_size})...")
        generated_count = 0
        while generated_count < num_total_samples:
            current_batch_size = min(chunk_size, num_total_samples - generated_count)
            batch_chromatograms = np.zeros((current_batch_size, signal_length), dtype=np.float32)
            batch_target_maps = np.zeros((current_batch_size, signal_length), dtype=np.float32)
            batch_peak_params_str_list = []

            for i in range(current_batch_size):
                X, Y_map, peak_params_list_item, _x_coords = generator_fn(**generator_config)
                batch_chromatograms[i, :] = X
                batch_target_maps[i, :] = Y_map
                batch_peak_params_str_list.append(json.dumps(peak_params_list_item))

            start_idx = generated_count;
            end_idx = generated_count + current_batch_size
            dset_chroms[start_idx:end_idx, :] = batch_chromatograms
            dset_maps[start_idx:end_idx, :] = batch_target_maps
            dset_peak_params[start_idx:end_idx] = batch_peak_params_str_list

            generated_count += current_batch_size
            if generated_count % (max(1, chunk_size * 2)) == 0 or generated_count == num_total_samples:
                print(f"  Generated and saved {generated_count}/{num_total_samples} samples...")
        print(f"Offline dataset creation complete: {filepath}")


# --- Example: Script to Generate/Test ---
if __name__ == '__main__':
    OFFLINE_DATA_CONFIG = {
        "filepath": "fsec_offline_dataset_y_corrected.h5",
        "num_total_samples": 500,
        "generator_fn": generate_synthetic_chromatogram_mixed_smooth_noise,
        "generator_config": {
            'length': 256, 'x_range': (0, 100), 'max_peaks': 3, 'min_peaks': 1,
            'amplitude_range': (0.8, 1.2), 'std_dev_range': (2.5, 5.5),
            'tau_range': (0.0, 1.0), 'peak_type_ratio': 0.5,
            'shoulder_peak_probability': 0.6, 'shoulder_amplitude_ratio_range': (0.1, 0.3),
            'shoulder_std_dev_ratio_range': (0.5, 0.9), 'shoulder_mu_offset_ratio_range': (-1.2, 1.2),
            'shoulder_tau_factor_range': (0.0, 0.5),
            'base_noise_level': 0.005, 'noise_smoothing_window_size': 9, 'noise_smoothing_polyorder': 2,
            # --- Y-Corrected Baseline Params for Test ---
            'baseline_linear_drift_variation': 0.08,
            # Max total change due to linear component (factor of typical_amplitude)
            'baseline_wobble_amplitude_factor': 0.04,  # Factor of typical_amplitude for wobble
            'baseline_wobble_periods_range': (0.4, 1.2),
            'baseline_y_corrected_min_level': -0.02
            # Allow baseline to dip slightly below zero (relative to typical_amplitude)
            # e.g. -0.02 means min baseline is -2% of typical_amplitude
        },
        "chunk_size": 100
    }

    if not os.path.exists(OFFLINE_DATA_CONFIG["filepath"]):
        print(f"Starting test offline dataset generation: {OFFLINE_DATA_CONFIG['filepath']}")
        create_large_offline_dataset(
            OFFLINE_DATA_CONFIG["filepath"], OFFLINE_DATA_CONFIG["num_total_samples"],
            OFFLINE_DATA_CONFIG["generator_fn"], OFFLINE_DATA_CONFIG["generator_config"],
            OFFLINE_DATA_CONFIG["chunk_size"]
        )
    else:
        print(f"Test offline dataset {OFFLINE_DATA_CONFIG['filepath']} already exists.")

    print("\nVisualizing some generated samples with Y-corrected baseline...")
    num_vis_samples = 3
    fig, axs = plt.subplots(num_vis_samples, 1, figsize=(12, 3 * num_vis_samples), sharex=True)
    if num_vis_samples == 1: axs = [axs]

    for i in range(num_vis_samples):
        X, Y_map, params_list, x_coords = generate_synthetic_chromatogram_mixed_smooth_noise(
            **OFFLINE_DATA_CONFIG["generator_config"]
        )
        axs[i].plot(x_coords, X, label="Final Signal (Y-corr Baseline + Noise)")
        true_signal_sum = np.zeros_like(x_coords)
        for p in params_list:
            tau_val = p.get('tau', 0.0) if p.get('tau') is not None else 0.0
            comp_peak = emg_peak_scipy(x_coords, p['A'], p['mu'], p['sigma'], tau_val) if (
                        p.get('type') == 'emg' or tau_val > 1e-5) else gaussian_peak_scipy(x_coords, p['A'], p['mu'],
                                                                                           p_sigma=p['sigma'])
            true_signal_sum += comp_peak
        axs[i].plot(x_coords, true_signal_sum, label="True Peaks Sum (No Baseline/Noise)", linestyle='--', alpha=0.7)

        # Reconstruct the generated baseline for plotting
        # This requires re-running the baseline part of the generator with same random state (hard)
        # OR, approximate by subtracting true peaks from signal before noise
        # For simplicity, let's plot X - true_signal_sum as an approximation of (Baseline + Noise)
        approx_baseline_plus_noise = X - true_signal_sum
        axs[i].plot(x_coords, approx_baseline_plus_noise, label="Approx. Baseline + Noise", linestyle=':', color='gray',
                    alpha=0.7)
        axs[i].axhline(0, color='black', linestyle='-', linewidth=0.5)  # Zero line
        axs[i].set_title(f"Sample {i + 1} with Y-Corrected Baseline Logic")
        axs[i].legend(fontsize='small');
        axs[i].grid(True, alpha=0.3)
    plt.xlabel("Elution Volume");
    plt.tight_layout();
    plt.show()
    print("Refined synthetic_data.py (Y-corrected baseline) script complete.")
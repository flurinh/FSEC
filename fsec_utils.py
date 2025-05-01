# fsec_utils.py

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.signal
from scipy.optimize import curve_fit
from scipy.special import erf
from itertools import combinations
from tqdm.auto import tqdm

# --- Data Loading --- (Unchanged from previous versions)
def load_fsec_data_combined(directory='input', include_filename=False, filename_column='filename'):
    """Loads and combines FSEC data from CSV files."""
    if not os.path.isdir(directory): raise FileNotFoundError(f"Error: Directory '{directory}' not found.")
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files: print(f"Warning: No CSV files found in '{directory}'."); return pd.DataFrame()
    all_data = []
    print(f"Found {len(csv_files)} CSV files in '{directory}'. Reading...")
    for file_path in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(file_path, on_bad_lines='warn')
            if df.empty: print(f"Warning: File '{file_path}' empty/unparseable. Skipping."); continue
            if include_filename: df[filename_column] = os.path.splitext(os.path.basename(file_path))[0]
            all_data.append(df)
        except Exception as e: print(f"Error processing '{file_path}': {e}. Skipping.")
    if not all_data: print("Warning: No data loaded."); return pd.DataFrame()
    try:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Successfully combined data from {len(all_data)} files.")
        if 'Timepoint' not in combined_df.columns: print("Warning: 'Timepoint' column missing.")
        return combined_df
    except Exception as e: print(f"Error during concatenation: {e}"); return pd.DataFrame()

# --- Signal Processing Utilities --- (Unchanged)
def calculate_durbin_watson(residuals):
    """Calculates the Durbin-Watson statistic."""
    if residuals is None or len(residuals) < 2: return 2.0
    residuals = np.asarray(residuals)[~np.isnan(residuals)]
    if len(residuals) < 2: return 2.0
    diff_residuals = np.diff(residuals); sum_sq_diff = np.sum(diff_residuals**2); sum_sq_res = np.sum(residuals**2)
    if sum_sq_res == 0: return 2.0
    dw = sum_sq_diff / sum_sq_res; n = len(residuals)
    return dw * (n / (n - 1)) if n > 1 else dw

def find_optimal_sg_window(signal, timepoints=None, polynomial_order=2, min_window=5, max_window=151):
    """Finds optimal SG window size using Durbin-Watson."""
    signal = np.asarray(signal); finite_mask = np.isfinite(signal)
    if not np.any(finite_mask): print("Warning: No finite values. Cannot find optimal SG window."); return max(5, min_window if min_window % 2 != 0 else min_window + 1)
    signal_finite = signal[finite_mask]; n_finite = len(signal_finite)
    min_req_points = polynomial_order + 2
    if n_finite < min_req_points or n_finite < min_window: print(f"Warning: Not enough finite points ({n_finite}). Using min window."); return max(5, min_window if min_window % 2 != 0 else min_window + 1)
    best_window = min_window if min_window % 2 != 0 else min_window + 1
    best_dw_dist = float('inf')
    max_window_adj = min(max_window if max_window % 2 != 0 else max_window - 1, n_finite if n_finite % 2 != 0 else n_finite - 1)
    max_window_adj = max(best_window, max_window_adj)
    for window_size in range(best_window, max_window_adj + 1, 2):
        if window_size <= polynomial_order: continue
        try:
            smoothed_signal = scipy.signal.savgol_filter(signal_finite, window_size, polynomial_order)
            residuals = signal_finite - smoothed_signal; dw = calculate_durbin_watson(residuals); dw_dist = abs(dw - 2.0)
            if dw_dist < best_dw_dist: best_dw_dist = dw_dist; best_window = window_size
        except ValueError as e: print(f"Skipping SG window {window_size}: {e}")
    return best_window

def snip_baseline(signal, window_size):
    """Applies SNIP baseline correction, handling NaNs via interpolation."""
    # (Code from previous version - unchanged)
    signal = np.asarray(signal); original_nan_mask = np.isnan(signal); signal_interp = np.copy(signal)
    if np.any(original_nan_mask):
        # print("Warning: NaNs detected in signal for SNIP. Interpolating linearly.") # Less verbose
        x_coords = np.arange(len(signal)); finite_mask = ~original_nan_mask
        if np.sum(finite_mask) >= 2:
            signal_interp[original_nan_mask] = np.interp(x_coords[original_nan_mask], x_coords[finite_mask], signal_interp[finite_mask])
            if np.isnan(signal_interp).any(): signal_interp = pd.Series(signal_interp).fillna(method='ffill').fillna(method='bfill').values
        else: signal_interp[original_nan_mask] = 0.0
    signal_safe = np.maximum(signal_interp, 1e-6); s_lls = np.log(np.log(np.sqrt(signal_safe + 1) + 1) + 1); s_lls_filt = np.copy(s_lls)
    half_window = max(1, int(window_size / 2))
    for m in range(1, half_window + 1):
        padded = np.pad(s_lls_filt, pad_width=m, mode='edge'); smoothed_padded = np.copy(padded)
        for i in range(m, len(padded) - m): smoothed_padded[i] = min(padded[i], (padded[i - m] + padded[i + m]) / 2)
        s_lls_filt = smoothed_padded[m:-m]
    baseline_interp = (np.exp(np.exp(s_lls_filt) - 1) - 1)**2 - 1
    baseline_interp = np.minimum(baseline_interp, signal_interp); baseline_interp = np.maximum(baseline_interp, 0)
    corrected_signal_interp = signal_interp - baseline_interp
    baseline = np.copy(baseline_interp); baseline[original_nan_mask] = np.nan
    corrected_signal = np.copy(corrected_signal_interp); corrected_signal[original_nan_mask] = np.nan
    return corrected_signal, baseline

def normalize_signal(signal, timepoints):
    """Performs baseline correction using SNIP."""
    # (Code from previous version - unchanged)
    sg_window_size = find_optimal_sg_window(signal, timepoints)
    print(f"Optimal SG window size for baseline/smoothing: {sg_window_size}")
    snip_window = sg_window_size
    corrected_signal, baseline = snip_baseline(signal, snip_window)
    corrected_signal = np.clip(corrected_signal, 0, None, where=~np.isnan(corrected_signal))
    baseline_finite = baseline[np.isfinite(baseline)]
    baseline_std = np.std(baseline_finite) if len(baseline_finite) > 0 else 0.0
    # print(f"Calculated Baseline Std Dev: {baseline_std:.4f}") # Moved print to analyze
    return corrected_signal, baseline, baseline_std, sg_window_size

# --- Peak and Shoulder Detection (Simplified Approach) ---

# NEW Simple Peak Finder
def find_peaks_abs_prominence(corrected_signal, timepoints, prominence, width):
    """
    Finds peaks on the corrected signal using absolute prominence and width.

    Args:
        corrected_signal (np.array): Signal after baseline correction (original scale).
        timepoints (np.array): Corresponding time points (for reporting).
        prominence (float): Minimum absolute prominence (in signal units).
        width (int): Minimum width of peaks in data points.

    Returns:
        tuple: (peak_indices, properties)
    """
    peak_indices = np.array([], dtype=int)
    properties = {}
    finite_mask = np.isfinite(corrected_signal)
    if not np.any(finite_mask): print("Warning: Corrected signal has no finite values."); return peak_indices, properties

    print(f"--- find_peaks_abs_prominence ---")
    print(f"  Using absolute prominence >= {prominence:.4f}")
    print(f"  Using width >= {width}")

    try:
        # Find peaks directly on the finite part of the corrected signal
        finite_indices = np.where(finite_mask)[0]
        peaks_rel, properties = scipy.signal.find_peaks(
            corrected_signal[finite_mask],
            prominence=prominence,
            width=width
        )
        peak_indices = finite_indices[peaks_rel] # Map back to original indices
    except Exception as e:
         print(f"Error during find_peaks_abs_prominence: {e}")

    print(f"  Found {len(peak_indices)} initial peaks.")
    if len(peak_indices) > 0:
         print(f"  Indices: {peak_indices}")
         # print(f"  Timepoints: {timepoints[peak_indices]}") # Can be verbose
         # print(f"  Prominences: {properties.get('prominences', 'N/A')}")
    return peak_indices, properties


# NEW Shoulder/Feature Detector based on Derivative Threshold
def detect_features_by_derivative(corrected_signal, timepoints, main_peak_indices,
                                 sg_window_size, shoulder_deriv_threshold,
                                 peak_width_for_prox_check, debug=False):
    """
    Detects potential shoulder features based on second derivative threshold.

    Args:
        corrected_signal (np.array): Baseline-corrected signal (original scale).
        timepoints (np.array): Corresponding time points.
        main_peak_indices (np.array): Indices of initially detected main peaks.
        sg_window_size (int): Window size for Savitzky-Golay derivative calculation.
        shoulder_deriv_threshold (float): Minimum absolute value of the negative second derivative
                                         peak height to consider a point as a candidate.
        peak_width_for_prox_check (int): Width used for main peak finding, helps define
                                        proximity tolerance for filtering.
        debug (bool): Print detailed debug info.

    Returns:
        np.array: Indices of potential shoulder candidates (excluding main peaks).
    """
    potential_shoulder_indices = np.array([], dtype=int)
    if corrected_signal is None or len(corrected_signal) < 5: return potential_shoulder_indices

    corrected_signal = np.asarray(corrected_signal)
    timepoints = np.asarray(timepoints)
    main_peak_indices_set = set(main_peak_indices) if main_peak_indices is not None else set()

    # 1. Calculate Second Derivative (on original scale corrected signal)
    deriv_window = max(5, sg_window_size if sg_window_size % 2 != 0 else sg_window_size + 1)
    if deriv_window >= len(corrected_signal):
        print(f"Warning: Deriv window {deriv_window} too large. Cannot detect shoulders."); return potential_shoulder_indices

    # Handle NaNs before derivative calculation
    signal_for_deriv = np.copy(corrected_signal)
    nan_mask_deriv = np.isnan(signal_for_deriv)
    if np.any(nan_mask_deriv):
        # print("Debug: Interpolating NaNs for derivative calculation.")
        x_coords_d = np.arange(len(signal_for_deriv))
        finite_mask_d = ~nan_mask_deriv
        if np.sum(finite_mask_d) >= 2:
             signal_for_deriv[nan_mask_deriv] = np.interp(x_coords_d[nan_mask_deriv], x_coords_d[finite_mask_d], signal_for_deriv[finite_mask_d])
             if np.isnan(signal_for_deriv).any(): signal_for_deriv = pd.Series(signal_for_deriv).fillna(method='ffill').fillna(method='bfill').values
        else: signal_for_deriv[nan_mask_deriv] = 0.0 # Fallback

    try:
        second_derivative = scipy.signal.savgol_filter(signal_for_deriv, deriv_window, 2, deriv=2)
    except ValueError as e:
        print(f"Error calculating second derivative: {e}. Cannot detect shoulders."); return potential_shoulder_indices

    neg_second_derivative = -second_derivative
    neg_second_derivative[~np.isfinite(neg_second_derivative)] = 0.0 # Handle potential NaNs/Infs from filter

    if debug:
        print("\n--- detect_features_by_derivative ---")
        print(f"  Using derivative window: {deriv_window}")
        print(f"  Negative 2nd Deriv range: {np.min(neg_second_derivative):.4f} / {np.max(neg_second_derivative):.4f}")
        print(f"  Using derivative height threshold >= {shoulder_deriv_threshold:.4f}")


    # 2. Find peaks in negative 2nd derivative exceeding the threshold
    deriv_candidate_indices = np.array([], dtype=int)
    deriv_peak_properties = {}
    if shoulder_deriv_threshold >= 0: # Only find if threshold is non-negative
        try:
            deriv_candidate_indices, deriv_peak_properties = scipy.signal.find_peaks(
                neg_second_derivative,
                height=shoulder_deriv_threshold # Use height parameter
                # width= ? # Optional: Add a minimum width for derivative peaks? max(1, int(peak_width_for_prox_check / 4)) ?
            )
        except Exception as e:
             print(f"Error finding peaks in negative 2nd derivative: {e}")
    else:
        print("Warning: shoulder_deriv_threshold is negative, skipping derivative peak finding.")


    if debug:
        print(f"  Found {len(deriv_candidate_indices)} candidates based on derivative height.")
        if len(deriv_candidate_indices) > 0:
            print(f"  Candidate indices: {deriv_candidate_indices}")
            # print(f"  Candidate derivative heights: {deriv_peak_properties.get('peak_heights', 'N/A')}") # 'heights' used in find_peaks

    # 3. Filter out candidates too close to main peaks
    peak_tolerance = max(2, int(peak_width_for_prox_check / 5))
    potential_shoulder_indices_list = []
    for idx in deriv_candidate_indices:
        is_main_peak = False
        for main_idx in main_peak_indices_set:
            if abs(idx - main_idx) <= peak_tolerance:
                is_main_peak = True
                break
        if not is_main_peak:
            potential_shoulder_indices_list.append(idx)

    potential_shoulder_indices = np.array(sorted(potential_shoulder_indices_list), dtype=int)

    if debug:
        print(f"  Found {len(potential_shoulder_indices)} potential shoulders after removing main peak proximity.")
        if len(potential_shoulder_indices) > 0:
             print(f"  Potential shoulder indices: {potential_shoulder_indices}")

    return potential_shoulder_indices


# NEW Simplified Main Analysis Function
def analyze_chromatogram_simplified(combined_df, column_name,
                                    peak_prominence=5.0, # Absolute units now
                                    peak_width=10,
                                    shoulder_deriv_threshold=0.01, # Absolute deriv units
                                    min_feature_height=1.0, # Absolute signal units
                                    plot=True, debug=False):
    """
    Analyzes chromatogram using absolute thresholds for peaks and derivative features.

    Args:
        combined_df (pd.DataFrame): Input data.
        column_name (str): Signal column name.
        peak_prominence (float): Minimum absolute prominence for main peaks (corrected signal units).
        peak_width (int): Minimum width for main peaks (points).
        shoulder_deriv_threshold (float): Minimum height of peak in negative 2nd derivative
                                         to be considered a shoulder candidate.
        min_feature_height (float): Minimum height on corrected signal for ANY feature
                                   (peak or shoulder) to be kept.
        plot (bool): Generate plots.
        debug (bool): Print debug info for shoulder detection.

    Returns:
        dict: Results dictionary or None on error.
    """
    if 'Timepoint' not in combined_df.columns: print("Error: 'Timepoint' column missing."); return None
    if column_name not in combined_df.columns: print(f"Error: Column '{column_name}' missing."); return None
    signal = combined_df[column_name].values; timepoints = combined_df['Timepoint'].values # Assuming time_col is defined globally or passed
    if len(signal) == 0 or np.all(np.isnan(signal)): print(f"Error: Signal '{column_name}' empty/all NaN."); return None
    print(f"\n--- Analyzing Column: {column_name} (Simplified Method) ---")

    # 1. Baseline Correction
    corrected_signal, baseline, baseline_std, sg_window_size = normalize_signal(signal, timepoints)
    print(f"  Baseline Std Dev (Original Scale): {baseline_std:.4f}")

    # 2. Find Initial Main Peaks (Absolute Prominence)
    initial_peak_indices, initial_peak_props = find_peaks_abs_prominence(
        corrected_signal, timepoints, prominence=peak_prominence, width=peak_width)

    # 3. Find Potential Shoulders (Absolute Derivative Threshold)
    potential_shoulder_indices = detect_features_by_derivative(
        corrected_signal=corrected_signal,
        timepoints=timepoints,
        main_peak_indices=initial_peak_indices,
        sg_window_size=sg_window_size,
        shoulder_deriv_threshold=shoulder_deriv_threshold,
        peak_width_for_prox_check=peak_width,
        debug=debug
    )

    # 4. Apply Minimum Feature Height Filter to BOTH Peaks and Shoulders
    final_peak_indices = []
    final_shoulder_indices = []

    print(f"--- Applying Minimum Feature Height Filter >= {min_feature_height:.4f} ---")
    # Filter initial peaks
    for idx in initial_peak_indices:
        if 0 <= idx < len(corrected_signal) and np.isfinite(corrected_signal[idx]) and corrected_signal[idx] >= min_feature_height:
            final_peak_indices.append(idx)
        elif debug: print(f"    - Peak at index {idx} rejected by height filter (height={corrected_signal[idx]:.4f})")

    # Filter potential shoulders
    for idx in potential_shoulder_indices:
        if 0 <= idx < len(corrected_signal) and np.isfinite(corrected_signal[idx]) and corrected_signal[idx] >= min_feature_height:
            # Double-check it wasn't already accepted as a peak (shouldn't happen if filtering in detect_features is correct, but safe)
            if idx not in final_peak_indices:
                 final_shoulder_indices.append(idx)
        elif debug: print(f"    - Shoulder candidate at index {idx} rejected by height filter (height={corrected_signal[idx]:.4f})")

    final_peak_indices = np.array(sorted(final_peak_indices), dtype=int)
    final_shoulder_indices = np.array(sorted(final_shoulder_indices), dtype=int)

    print(f"  Final Peaks: {len(final_peak_indices)}")
    print(f"  Final Shoulders: {len(final_shoulder_indices)}")

    # 5. Plotting
    if plot:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(timepoints, signal, label='Original Signal', color='gray', alpha=0.6, lw=1.5)
            ax1.plot(timepoints, corrected_signal, label='Corrected Signal', color='blue', lw=1.5)
            ax1.plot(timepoints, baseline, label=f'Baseline (SNIP, win={sg_window_size})', color='orange', linestyle='--', lw=1.5)
            ax1.axhline(min_feature_height, color='red', linestyle=':', alpha=0.5, lw=1, label=f'Min Height ({min_feature_height:.2f})') # Show height threshold

            n_peaks = len(final_peak_indices)
            if n_peaks > 0: ax1.plot(timepoints[final_peak_indices], corrected_signal[final_peak_indices], "x", color='red', markersize=10, mew=2, label=f'Peaks ({n_peaks})', linestyle='None')
            else: ax1.plot([],[], "x", color='red', markersize=10, mew=2, label='Peaks (0)')
            n_shoulders = len(final_shoulder_indices)
            if n_shoulders > 0:
                ax1.plot(timepoints[final_shoulder_indices], corrected_signal[final_shoulder_indices], "o", color='green', markersize=8, label=f'Shoulders ({n_shoulders})', alpha=0.9, markerfacecolor='none', mew=2, linestyle='None')
            else: ax1.plot([],[], "o", color='green', markersize=8, markerfacecolor='none', mew=2, label='Shoulders (0)')

            ax1.set_title(f'Simplified Peak/Shoulder Detection ({column_name})'); ax1.set_ylabel('Signal')
            ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.4, linestyle=':')
            finite_y = np.concatenate([sig[np.isfinite(sig)] for sig in [signal, corrected_signal, baseline] if sig is not None])
            if len(finite_y)>0: y_min, y_max = np.min(finite_y), np.max(finite_y); y_buffer = (y_max - y_min) * 0.05 if (y_max - y_min)>0 else 0.1; ax1.set_ylim(bottom=min(y_min - y_buffer, -y_buffer), top=y_max + y_buffer) # Ensure 0 visible if data positive

            # Bottom plot: Negative Second derivative
            deriv_window = max(5, sg_window_size if sg_window_size % 2 != 0 else sg_window_size + 1)
            if deriv_window < len(corrected_signal):
                 signal_for_deriv = np.copy(corrected_signal); nan_mask_deriv = np.isnan(signal_for_deriv) # Re-handle NaNs locally for derivative plot
                 if np.any(nan_mask_deriv):
                     x_coords_d = np.arange(len(signal_for_deriv)); finite_mask_d = ~nan_mask_deriv
                     if np.sum(finite_mask_d) >= 2: signal_for_deriv[nan_mask_deriv] = np.interp(x_coords_d[nan_mask_deriv], x_coords_d[finite_mask_d], signal_for_deriv[finite_mask_d]);
                     if np.isnan(signal_for_deriv).any(): signal_for_deriv = pd.Series(signal_for_deriv).fillna(method='ffill').fillna(method='bfill').values
                     else: signal_for_deriv[nan_mask_deriv] = 0.0
                 try:
                     second_derivative = scipy.signal.savgol_filter(signal_for_deriv, deriv_window, 2, deriv=2)
                     neg_second_derivative = -second_derivative
                     neg_second_derivative[~np.isfinite(neg_second_derivative)] = 0.0 # Handle Inf/NaN from filter edges
                     ax2.plot(timepoints, neg_second_derivative, label='-2nd Deriv', color='purple', lw=1.5)
                     ax2.axhline(shoulder_deriv_threshold, color='magenta', linestyle=':', alpha=0.7, lw=1, label=f'Deriv Thresh ({shoulder_deriv_threshold:.3f})')
                     # Mark locations (optional, can get cluttered)
                     # ax2.plot(timepoints[final_peak_indices], neg_second_derivative[final_peak_indices], "x", color='red', markersize=8, mew=2, label='Peak Locs', linestyle='None')
                     # ax2.plot(timepoints[final_shoulder_indices], neg_second_derivative[final_shoulder_indices], "o", color='green', markerfacecolor='none', mew=2, markersize=6, label='Shoulder Locs', linestyle='None')
                 except Exception as e: print(f"Error plotting derivative: {e}"); ax2.text(0.5, 0.5, "Error plotting derivative", ha='center', va='center', transform=ax2.transAxes)
            else: ax2.text(0.5, 0.5, "Signal too short for 2nd derivative plot", ha='center', va='center', transform=ax2.transAxes)
            time_col_name = combined_df.columns[0] # Get actual time column name
            ax2.set_xlabel(f'{time_col_name}'); ax2.set_ylabel('Neg. 2nd Deriv'); ax2.axhline(0, color='black', linewidth=0.5, linestyle='--'); ax2.grid(True, alpha=0.4, linestyle=':')
            ax2.legend(loc='upper right'); ax2.autoscale(enable=True, axis='y', tight=True); plt.tight_layout(pad=0.5); plt.show()
        except Exception as e: print(f"Error during plotting: {e}")

    # 6. Return Results
    results = {
        'column_name': column_name, 'timepoints': timepoints, 'original_signal': signal,
        'baseline': baseline, 'corrected_signal': corrected_signal, 'baseline_std': baseline_std,
        'peak_indices': final_peak_indices, #'initial_peak_properties': initial_peak_props, # Maybe too verbose
        'shoulder_indices': final_shoulder_indices,
        'sg_window_size': sg_window_size,
        # Store parameters used
        'params': {'peak_prominence': peak_prominence, 'peak_width': peak_width,
                   'shoulder_deriv_threshold': shoulder_deriv_threshold, 'min_feature_height': min_feature_height}
    }
    print(f"--- Analysis Finished for: {column_name} (Simplified Method) ---")
    return results

# --- Keep Chromatogram Class ---
# (It will need updating if used with the new simplified analysis, but leave as is for now)
class Chromatogram:
     """Class for handling chromatography data (using older methods)."""
     # (Code from previous version - unchanged)
     def __init__(self, df, cols={'time': 'Timepoint', 'signal': None}):
        if not isinstance(df, pd.DataFrame) or df.empty: raise ValueError("Input must be a non-empty pandas DataFrame.")
        self.df_orig = df.copy(); self.df = df.copy()
        if cols.get('time') in self.df.columns: self.time_col = cols['time']
        else: raise ValueError(f"Time column '{cols.get('time')}' not found in DataFrame columns: {list(self.df.columns)}")
        sig_col_name = cols.get('signal')
        if sig_col_name is None:
            signal_cols = [col for col in self.df.columns if col != self.time_col and pd.api.types.is_numeric_dtype(self.df[col])]
            if not signal_cols: raise ValueError("No potential numeric signal columns found besides time column.")
            self.signal_col = signal_cols[0]; print(f"Warning: No signal column specified. Using: '{self.signal_col}'")
        elif sig_col_name in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[sig_col_name]): raise ValueError(f"Signal column '{sig_col_name}' is not numeric.")
            self.signal_col = sig_col_name
        else: raise ValueError(f"Signal column '{sig_col_name}' not found in DataFrame columns: {list(self.df.columns)}")
        self.time = self.df[self.time_col].values; self.signal = self.df[self.signal_col].values
        if np.isnan(self.time).any(): print(f"Warning: Time column '{self.time_col}' contains NaN values.")
        if np.isnan(self.signal).any(): print(f"Warning: Signal column '{self.signal_col}' contains NaN values.")
        self.reset_processing()

     def reset_processing(self):
        self.baseline = None; self.corrected_signal = None; self.peaks = None
        self.peak_properties = None; self.peak_fits = None; self.reconstructed_signal = None
        self.fit_assessment = None; self.sg_window_size = None; self.corrected_signal_range = None

     def crop(self, time_range):
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2: raise ValueError("time_range must be [min_time, max_time]")
        min_t, max_t = time_range
        if not (isinstance(min_t,(int,float)) and isinstance(max_t,(int,float))): raise ValueError("time_range values must be numeric")
        mask = (self.df_orig[self.time_col] >= min_t) & (self.df_orig[self.time_col] <= max_t)
        self.df = self.df_orig[mask].copy()
        self.time = self.df[self.time_col].values; self.signal = self.df[self.signal_col].values
        self.reset_processing()
        print(f"Cropped chromatogram to time range: [{min_t:.3f}, {max_t:.3f}]. Data points: {len(self.time)}")
        if len(self.time) == 0: print("Warning: Crop resulted in zero data points.")
        return self

     def show(self, title=None, show_corrected=True, show_peaks=True, show_reconstructed=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(self.time) == 0: ax.text(0.5, 0.5, "No data.", ha='center', va='center'); ax.set_title("Empty Chromatogram"); return fig, ax
        ax.plot(self.time, self.signal, color='gray', label='Original Signal', alpha=0.8)
        if show_corrected and self.corrected_signal is not None:
            ax.plot(self.time, self.baseline, color='orange', linestyle='--', label='Baseline')
            ax.plot(self.time, self.corrected_signal, color='blue', label='Corrected Signal')
        if show_reconstructed and self.reconstructed_signal is not None:
            ax.plot(self.time, self.reconstructed_signal, color='purple', linestyle=':', label='Reconstructed Signal')
        if show_peaks and self.peaks is not None and self.corrected_signal is not None:
             valid_peak_indices = self.peaks[(self.peaks >= 0) & (self.peaks < len(self.time))]
             n_peaks_plot = len(valid_peak_indices)
             if n_peaks_plot > 0: ax.plot(self.time[valid_peak_indices], self.corrected_signal[valid_peak_indices], 'x', color='red', markersize=10, mew=2, label=f'Detected Peaks ({n_peaks_plot})')
             else: ax.plot([], [], 'x', color='red', markersize=10, mew=2, label='Detected Peaks (0)')
        ax.set_xlabel(self.time_col); ax.set_ylabel(self.signal_col)
        ax.set_title(title if title else f"Chromatogram: {self.signal_col}")
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
        return fig, ax

     def correct_baseline(self, window_size=None):
        if len(self.signal) == 0: print("Warning: Cannot correct baseline on empty signal."); return self
        if window_size is None: self.sg_window_size = find_optimal_sg_window(self.signal, self.time); print(f"Using optimal SG window for baseline: {self.sg_window_size}")
        else: window_size = int(window_size); self.sg_window_size = max(5, window_size if window_size % 2 != 0 else window_size + 1)
        self.corrected_signal, self.baseline = snip_baseline(self.signal, self.sg_window_size)
        self.corrected_signal = np.clip(self.corrected_signal, 0, None, where=~np.isnan(self.corrected_signal))
        finite_corr = self.corrected_signal[np.isfinite(self.corrected_signal)]; self.corrected_signal_range = np.ptp(finite_corr) if len(finite_corr) > 0 else 0.0
        print("Baseline correction applied.")
        return self

     def find_peaks(self, prominence=5.0, width=10): # Changed default prominence interpretation
         """Finds peaks using ABSOLUTE prominence."""
         if self.corrected_signal is None: print("Baseline correction not performed. Running correct_baseline first."); self.correct_baseline()
         if len(self.corrected_signal) == 0: print("Warning: Cannot find peaks on empty corrected signal."); self.peaks = np.array([], dtype=int); self.peak_properties = {}; return self
         # Use the simple absolute prominence finder
         self.peaks, self.peak_properties = find_peaks_abs_prominence(
              self.corrected_signal, self.time, prominence=prominence, width=width)
         print(f"Found {len(self.peaks)} peaks.")
         return self

     # --- fit_peaks and assess_fit methods remain the same as previous version ---
     # --- They operate on self.peaks found by the (now absolute) find_peaks method ---
     def fit_peaks(self, prominence=0.05, width=10, buffer=0.1):
        """Fits peaks individually using Gaussian models (simple method)."""
        if self.peaks is None: print("Peaks not found. Running find_peaks first."); self.find_peaks(prominence=prominence, width=width) # Using default absolute prominence here now
        if self.corrected_signal is None: print("Error: Corrected signal not available for fitting."); return self
        if len(self.corrected_signal) == 0 : print("Warning: Corrected signal is empty, cannot fit peaks."); return self
        if self.peaks is None or len(self.peaks) == 0: print("No peaks detected to fit."); self.peak_fits = []; self.reconstructed_signal = np.zeros_like(self.time, dtype=float); return self
        def gaussian(x, amp, mu, sigma): sigma = max(1e-6, sigma); return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        self.peak_fits = []
        print(f"Fitting {len(self.peaks)} detected peaks individually with Gaussians...")
        time_diff = np.mean(np.diff(self.time)) if len(self.time) > 1 else 0.01

        for peak_idx in tqdm(self.peaks, desc="Fitting individual peaks"):
            fit_result = {'index': peak_idx, 'time': np.nan, 'amp': np.nan,'mu': np.nan, 'sigma': np.nan, 'area': np.nan, 'fit_success': False}
            if not (0 <= peak_idx < len(self.time)): print(f"Warning: Skipping invalid peak index {peak_idx}"); self.peak_fits.append(fit_result); continue
            fit_result['time'] = self.time[peak_idx]; peak_height = self.corrected_signal[peak_idx];
            if not np.isfinite(peak_height): peak_height = 0
            left_time = fit_result['time'] - buffer; right_time = fit_result['time'] + buffer
            fit_mask = (self.time >= left_time) & (self.time <= right_time) & np.isfinite(self.corrected_signal)
            x_data = self.time[fit_mask]; y_data = self.corrected_signal[fit_mask]
            if len(x_data) < 3: print(f"Warning: Skipping peak at t={fit_result['time']:.3f} - not enough finite points ({len(x_data)})"); self.peak_fits.append(fit_result); continue
            amp_guess = peak_height if peak_height > 0 else 1e-3; mu_guess = fit_result['time']; sigma_guess = buffer / 3.0
            if self.peak_properties and 'widths' in self.peak_properties:
                 try: # Simplified width estimation logic
                      prop_indices = np.where(self.peaks == peak_idx)[0]
                      if len(prop_indices) > 0 and prop_indices[0] < len(self.peak_properties['widths']):
                           peak_width_points = self.peak_properties['widths'][prop_indices[0]]
                           peak_width_time = peak_width_points * time_diff
                           if peak_width_time > 1e-6: sigma_guess = max(peak_width_time / 2.355, time_diff * 1.5)
                 except Exception: pass
            p0 = [amp_guess, mu_guess, sigma_guess]; bounds = ([0, left_time, time_diff/2.0], [amp_guess * 5 + 1e-3, right_time, buffer * 5])
            try:
                popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000, bounds=bounds)
                fit_result.update({'amp': popt[0],'mu': popt[1],'sigma': popt[2], 'area': popt[0] * popt[2] * np.sqrt(2 * np.pi),'fit_success': True})
                if popt[2] < (time_diff / 2.0): pass # Potentially mark as uncertain if sigma too small?
            except Exception as e: print(f"Warning: Fit failed for peak at t={fit_result['time']:.3f}. Error: {e}")
            self.peak_fits.append(fit_result)
        self.reconstructed_signal = np.zeros_like(self.time, dtype=float)
        for pf in self.peak_fits:
             if pf.get('fit_success', False) and not np.isnan(pf['amp']): self.reconstructed_signal += gaussian(self.time, pf['amp'], pf['mu'], pf['sigma'])
        return self

     def assess_fit(self, tolerance=0.10, print_report=True, use_ansi_colors=False):
        """Assess quality of individual peak fitting."""
        # (assess_fit code remains unchanged from previous version)
        if self.peaks is None or self.corrected_signal is None or self.peak_fits is None: print("Error: Must run find_peaks() and fit_peaks() before assessing fit."); return None
        if len(self.corrected_signal) == 0: print("Warning: Corrected signal empty."); return None
        if self.reconstructed_signal is None: print("Error: Reconstructed signal not calculated."); return None
        valley_indices = np.array([], dtype=int)
        if np.any(np.isfinite(self.corrected_signal)):
            try:
                finite_corr_mask = np.isfinite(self.corrected_signal); finite_indices_corr = np.where(finite_corr_mask)[0]
                if len(finite_indices_corr) > 5:
                    valley_indices_rel, _ = scipy.signal.find_peaks(-self.corrected_signal[finite_corr_mask], prominence=0.01 * np.ptp(self.corrected_signal[finite_corr_mask]), width=3)
                    valley_indices = finite_indices_corr[valley_indices_rel]
            except Exception as e: print(f"Warning: Could not find valleys: {e}")
        peak_indices_valid = self.peaks[(self.peaks >=0) & (self.peaks < len(self.time))]
        boundaries = np.sort(np.concatenate(([0], peak_indices_valid, valley_indices, [len(self.time) - 1])))
        unique_boundaries = np.unique(boundaries[boundaries >= 0])
        regions = []
        if len(unique_boundaries) > 1:
            for i in range(len(unique_boundaries) - 1):
                start_idx = unique_boundaries[i]; end_idx = unique_boundaries[i+1]
                if start_idx >= end_idx: continue
                is_peak_region = any(start_idx <= pk_idx <= end_idx for pk_idx in peak_indices_valid)
                regions.append({'start_idx': start_idx, 'end_idx': end_idx, 'type': 'peak' if is_peak_region else 'interpeak'})
        else: print("Warning: Could not define valid regions for assessment.")
        results = []
        def r_score(sig_a, recon_a):
            if not (np.isfinite(sig_a) and np.isfinite(recon_a)): return np.nan
            return 1.0 if abs(sig_a) < 1e-9 and abs(recon_a) < 1e-9 else (np.inf * np.sign(recon_a) if abs(sig_a) < 1e-9 else recon_a / sig_a)
        for i, region in enumerate(regions):
            start_idx, end_idx = region['start_idx'], region['end_idx']; region_time = self.time[start_idx:end_idx+1]
            finite_mask_region = np.isfinite(self.corrected_signal[start_idx:end_idx+1])
            region_signal_finite = self.corrected_signal[start_idx:end_idx+1][finite_mask_region]; region_reconstructed_finite = self.reconstructed_signal[start_idx:end_idx+1][finite_mask_region]
            region_time_finite = region_time[finite_mask_region]
            if len(region_time_finite) < 3: continue
            try:
                signal_area = np.trapz(region_signal_finite, region_time_finite); reconstructed_area = np.trapz(region_reconstructed_finite, region_time_finite)
                r = r_score(signal_area, reconstructed_area); residual = region_signal_finite - region_reconstructed_finite
                rmse = np.sqrt(np.mean(residual**2)) if len(residual) > 0 else 0.0
                results.append({'window_id': i + 1, 'time_start': region_time_finite[0], 'time_end': region_time_finite[-1],'signal_area': signal_area, 'inferred_area': reconstructed_area, 'reconstruction_score (R)': r, 'rmse': rmse, 'window_type': region['type']})
            except Exception as e: print(f"Warning: Could not calculate metrics for region {i+1}: {e}")
        if not results: print("Warning: No valid regions found for assessment."); return None
        df_results = pd.DataFrame(results); df_results['applied_tolerance'] = tolerance; df_results['status'] = 'needs review'
        peak_mask = df_results['window_type'] == 'peak'; interpeak_mask = df_results['window_type'] == 'interpeak'
        df_results.loc[peak_mask, 'status'] = np.where((df_results.loc[peak_mask, 'reconstruction_score (R)'] >= 1 - tolerance) & (df_results.loc[peak_mask, 'reconstruction_score (R)'] <= 1 + tolerance), 'valid', 'failed (R-score)')
        finite_corrected_signal = self.corrected_signal[np.isfinite(self.corrected_signal)]; finite_time = self.time[np.isfinite(self.corrected_signal)]
        total_signal_area = np.trapz(finite_corrected_signal, finite_time) if len(finite_time)>1 else 0.0;
        if abs(total_signal_area) < 1e-9: total_signal_area = 1.0
        df_results.loc[interpeak_mask, 'status'] = np.where((np.abs(df_results.loc[interpeak_mask, 'inferred_area']) < 0.05 * abs(total_signal_area)) & (np.abs(df_results.loc[interpeak_mask, 'reconstruction_score (R)']) < 2.0), 'low signal (ok)', 'failed (interpeak fit)')
        df_results.loc[interpeak_mask & (np.abs(df_results['inferred_area']) < 0.01 * abs(total_signal_area)) & ~np.isfinite(df_results['reconstruction_score (R)']), 'status'] = 'low signal (ok)'
        self.fit_assessment = df_results
        if print_report: # (Report printing logic unchanged)
            CLR_OK = "\033[1m\033[42m\030m" if use_ansi_colors else ""; CLR_FAIL = "\033[1m\033[41m\037m" if use_ansi_colors else ""; CLR_WARN = "\033[1m\033[43m\030m" if use_ansi_colors else ""; CLR_END = "\033[0m" if use_ansi_colors else ""
            print("\n-------------------Chromatogram Reconstruction Report Card----------------------\n"); print("(Assessment based on individual Gaussian fits - may be inaccurate for overlaps)")
            print("\nReconstruction of Peak Regions\n============================\n"); peak_report = df_results[df_results['window_type'] == 'peak']
            if peak_report.empty: print("No peak regions assessed.")
            else:
                for _, row in peak_report.iterrows(): t_range = f"t: {row['time_start']:.3f}-{row['time_end']:.3f}"; r_val = row['reconstruction_score (R)']; status = row['status']; print(f"{CLR_OK if status == 'valid' else CLR_FAIL}{'A+' if status=='valid' else 'F'}, {'Success' if status=='valid' else 'Failed'}: Peak Window {row['window_id']} ({t_range}) R={r_val:.3f}{CLR_END}\n");
            print("\nReconstruction of Interpeak Regions\n=================================\n"); interpeak_report = df_results[df_results['window_type'] == 'interpeak']
            if interpeak_report.empty: print("No interpeak regions assessed.")
            else:
                 for _, row in interpeak_report.iterrows(): t_range = f"t: {row['time_start']:.3f}-{row['time_end']:.3f}"; r_val = row['reconstruction_score (R)']; inferred_a = row['inferred_area']; status = row['status']; print(f"{CLR_OK if status == 'low signal (ok)' else CLR_WARN}{'OK' if status=='low signal (ok)' else 'C'}, {'Good' if status=='low signal (ok)' else 'Needs Review'}: Interpeak Window {row['window_id']} ({t_range}) R={r_val:.3f}, Inferred Area={inferred_a:.3e}{CLR_END}\n");
            print("\n--------------------------------------------------------------------------------\n")
        return df_results



# --- Example Usage Block --- (Updated)
if __name__ == '__main__':
    print("--- Running fsec_utils.py in test mode (Simplified Analysis) ---")
    time_col = 'Timepoint' # Define time column name

    # Create dummy data if 'input' directory doesn't exist or is empty
    if not os.path.exists('input') or not glob.glob('input/*.csv'):
        print("Creating dummy input data...")
        if not os.path.exists('input'): os.makedirs('input')
        time = np.linspace(0, 10, 500); peak1 = 10 * np.exp(-(time - 3)**2 / (2 * 0.5**2))
        baseline_dummy = 2 + 0.1 * time + np.random.rand(500) * 0.5; signal1 = peak1 + baseline_dummy
        df1 = pd.DataFrame({time_col: time, 'Sample_A': signal1}); df1.to_csv('input/sample_A.csv', index=False)
        peak2 = 8 * np.exp(-(time - 5.5)**2 / (2 * 0.4**2)); peak3 = 6 * np.exp(-(time - 6.5)**2 / (2 * 0.6**2))
        shoulder = 2 * np.exp(-(time - 5.0)**2 / (2 * 0.2**2)); signal2 = peak2 + peak3 + shoulder + baseline_dummy + 1
        df2 = pd.DataFrame({time_col: time, 'Sample_B': signal2}); df2.to_csv('input/sample_B.csv', index=False)
        print("Dummy data created: sample_A.csv, sample_B.csv")

    print("\nTesting Data Loading..."); combined = load_fsec_data_combined(directory='input')
    if not combined.empty:
        print(f"Loaded combined data shape: {combined.shape}"); print(f"Columns: {list(combined.columns)}")

        # --- Test Simplified Analysis on Sample B ---
        if 'Sample_B' in combined.columns:
            print("\nTesting analyze_chromatogram_simplified on Sample_B...")
            # Adjust thresholds based on expected absolute values of dummy data
            results_b = analyze_chromatogram_simplified(
                combined, 'Sample_B',
                peak_prominence=1.0,   # Expect peaks > 1 unit high
                peak_width=5,
                shoulder_deriv_threshold=0.5, # Requires significant curvature change
                min_feature_height=0.5,     # Feature must be > 0.5 units high
                plot=True, debug=True)
            if results_b:
                print("\nSimplified Analysis Results (Sample_B):"); print(f"  SG Window: {results_b['sg_window_size']}")
                print(f"  Peaks Found: {len(results_b['peak_indices'])} at times {results_b['timepoints'][results_b['peak_indices']] if len(results_b['peak_indices'])>0 else '[]'}")
                print(f"  Shoulders Found: {len(results_b['shoulder_indices'])} at times {results_b['timepoints'][results_b['shoulder_indices']] if len(results_b['shoulder_indices'])>0 else '[]'}")
    else: print("Could not load data for testing.")
    print("\n--- fsec_utils.py test finished ---")
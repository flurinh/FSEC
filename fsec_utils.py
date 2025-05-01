import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.signal
from scipy.optimize import curve_fit
from scipy.special import erf
from itertools import combinations


def load_fsec_data_combined(directory='input', include_filename=False, filename_column='filename'):
    """
    Loads FSEC data from all CSV files in a specified directory and combines them
    into a single DataFrame.

    Args:
        directory (str, optional): The directory containing the CSV files.
            Defaults to 'input'.
        include_filename (bool, optional): Whether to include a column with the
            original filename. Defaults to False.
        filename_column (str, optional): The name of the column to store
            filenames (if include_filename is True). Defaults to 'filename'.

    Returns:
        pandas.DataFrame: A single DataFrame containing all the FSEC data.
            Returns an empty DataFrame if no CSV files are found.
        Raises FileNotFoundError if the input directory doesn't exist.

    """

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Error: Directory '{directory}' not found.")

    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    if not csv_files:
        print(f"Warning: No CSV files found in '{directory}'.")
        return pd.DataFrame()  # Return an empty DataFrame

    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if include_filename:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                df[filename_column] = filename  # Add filename as a column
            all_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{file_path}' is empty. Skipping.")
        except pd.errors.ParserError:
            print(f"Error: Could not parse the file '{file_path}'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file_path}': {e}. Skipping.")

    if not all_data:  # Check if all_data is empty (e.g., all files were skipped)
        print("Warning: No data could be loaded from any files.")
        return pd.DataFrame()

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def calculate_durbin_watson(residuals):
    """Calculates the Durbin-Watson statistic."""
    diff_residuals = np.diff(residuals)
    dw = np.sum(diff_residuals**2) / np.sum(residuals**2)
    return dw * (len(residuals) / (len(residuals) - 1))


def find_optimal_sg_window(signal, timepoints, polynomial_order=2, min_window=5, max_window=151):
    """Finds optimal Savitzky-Golay window size (Durbin-Watson)."""
    best_window = min_window
    best_dw = 0
    for window_size in range(min_window, max_window + 1, 2):
        if window_size >= len(signal):
            break
        smoothed_signal = scipy.signal.savgol_filter(signal, window_size, polynomial_order)
        residuals = signal - smoothed_signal  # Residuals from *original* signal
        dw = calculate_durbin_watson(residuals)
        if abs(dw - 2) < abs(best_dw - 2):
            best_dw = dw
            best_window = window_size
    return best_window


def snip_baseline(signal, window_size):
    """Applies the SNIP baseline correction algorithm."""
    s_lls = np.log(np.log(np.sqrt(signal + 1) + 1) + 1)
    s_lls_filt = np.copy(s_lls)
    for m in range(1, window_size + 1):
        for i in range(m, len(s_lls_filt) - m):
            s_lls_filt[i] = min(s_lls_filt[i], (s_lls_filt[i - m] + s_lls_filt[i + m]) / 2)
    baseline = (np.exp(np.exp(s_lls_filt) - 1) - 1)**2 - 1
    corrected_signal = signal - baseline
    return corrected_signal, baseline


def normalize_signal(signal, timepoints):
    """
    Performs baseline correction and returns the corrected signal, baseline,
    and optimal SG window size.  This is Step 1.  Crucially, this function
    *returns* the baseline.  The corrected signal has the baseline subtracted.
    """
    sg_window_size = find_optimal_sg_window(signal, timepoints)
    print(f"Optimal SG window size: {sg_window_size}")
    corrected_signal, baseline = snip_baseline(signal, sg_window_size)
    return corrected_signal, baseline, sg_window_size


def find_main_peaks(corrected_signal, timepoints, prominence, width, sg_window_size):
    """
    Finds main peaks using prominence and width on the *corrected* signal.
    This is Step 2.  The corrected signal is normalized for peak finding.
    """
    # Normalize the *corrected* signal for peak finding.
    normalized_corrected_signal = (corrected_signal - np.min(corrected_signal)) / (np.max(corrected_signal) - np.min(corrected_signal))
    peak_indices, _ = scipy.signal.find_peaks(normalized_corrected_signal, prominence=prominence, width=width)
    return peak_indices, normalized_corrected_signal # Return normalized signal


def detect_shoulders(normalized_corrected_signal, timepoints, peak_indices, sg_window_size, baseline, shoulder_curvature, width):
    """
    Detects shoulders in the chromatogram after initial peak finding.

    Args:
        normalized_corrected_signal: Baseline-corrected and normalized signal.
        timepoints: Corresponding time values.
        peak_indices: Indices of the main peaks (found in Step 2).
        sg_window_size: Optimal Savitzky-Golay window size.
        shoulder_curvature: Minimum curvature (prominence for 2nd deriv. peaks).
        baseline: the calculated baseline
        width: Minimum width parameter for peak detection

    Returns:
        shoulder_indices: NumPy array of shoulder indices.
    """

    # 1. Calculate Second Derivative
    second_derivative = scipy.signal.savgol_filter(normalized_corrected_signal, window_length=sg_window_size, polyorder=2, deriv=2)

    # 2. Estimate Noise in Second Derivative (thrsd)
    h_values = []
    for i in range(1, len(second_derivative) - 1):
        h = abs(second_derivative[i] - (second_derivative[i-1] + second_derivative[i+1]) / 2)
        h_values.append(h)
    noise_estimate_sd = np.median(h_values)
    thrsd = 5 * noise_estimate_sd  # Threshold on second derivative

    # 3. Find Potential Shoulders (negative peaks in 2nd deriv)
    shoulder_peaks_neg, _ = scipy.signal.find_peaks(-second_derivative, prominence=thrsd, width=width/4)

    # 4. Signal Height Thresholds
    thrh1 = 3 * np.std(baseline)  # Dynamic threshold based on baseline noise
    thrh2 = 0  # Minimum signal height (can be adjusted)

    shoulder_indices = []

    # 5. find valleys
    valley_indices, _ = scipy.signal.find_peaks(-normalized_corrected_signal, prominence=0.1, width=width)

    # 6. Filter Shoulders
    for shoulder_idx in shoulder_peaks_neg:
        is_valid_shoulder = False

        # Check Signal Height
        if normalized_corrected_signal[shoulder_idx] >= thrh1 and normalized_corrected_signal[shoulder_idx] >= thrh2:

            # Find Nearest Peak and Valley
            nearest_peak = None
            nearest_valley = None
            min_peak_dist = float('inf')
            min_valley_dist = float('inf')

            for peak_idx in peak_indices:
                dist = abs(shoulder_idx - peak_idx)
                if dist < min_peak_dist:
                    min_peak_dist = dist
                    nearest_peak = peak_idx

            for valley_idx in valley_indices:
                dist = abs(shoulder_idx - valley_idx)
                if dist < min_valley_dist:
                    min_valley_dist = dist
                    nearest_valley = valley_idx

            # Peak-Valley-Shoulder Constraint
            if nearest_peak is not None and nearest_valley is not None:
                if (nearest_peak < shoulder_idx < nearest_valley) or \
                   (nearest_valley < shoulder_idx < nearest_peak):
                    is_valid_shoulder = True

        # Add if valid, not a duplicate, and not already a peak
        if is_valid_shoulder and shoulder_idx not in shoulder_indices and shoulder_idx not in peak_indices:
            shoulder_indices.append(shoulder_idx)

    return np.array(shoulder_indices, dtype=int)


def find_peaks_with_prominence(combined_df, column_name, prominence=0.02, baseline_window=10, plot=True):
    """
    Performs baseline correction and finds peaks based on prominence for a single column.

    Args:
        combined_df (pd.DataFrame): The combined DataFrame.
        column_name (str): The name of the column to process.
        prominence (float): The minimum prominence for peak detection.
        baseline_window (int): The window size for baseline correction.
        plot (bool): Whether to generate a plot.

    Returns:
        tuple: (peak_indices, corrected_signal, baseline, normalized_signal) or (None, None, None, None) if errors.
               peak_indices is a numpy array of integer indices.  corrected_signal, baseline, and normalized_signal
               are numpy arrays of floats.
    """
    if 'Timepoint' not in combined_df.columns:
        print("Error: 'Timepoint' column not found.")
        return None, None, None, None
    if column_name not in combined_df.columns:
        print(f"Error: Column '{column_name}' not found.")
        return None, None, None, None

    signal = combined_df[column_name].values
    timepoints = combined_df['Timepoint'].values

    # Baseline Correction
    corrected_signal, baseline = snip_baseline(signal, baseline_window)

    # Normalize the *original* signal for prominence calculation (important!)
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Find Peaks on the *corrected* signal
    peak_indices, properties = scipy.signal.find_peaks(corrected_signal, prominence=prominence)

    # --- Debugging Stats ---
    print(f"--- Stats for column: {column_name} ---")
    print(f"  Original signal range: {np.min(signal):.2f} - {np.max(signal):.2f}")
    print(f"  Corrected signal range: {np.min(corrected_signal):.2f} - {np.max(corrected_signal):.2f}")
    print(f"  Number of detected peaks: {len(peak_indices)}")
    if len(peak_indices) > 0:
        print(f"  Peak prominences: {properties['prominences']}") # Key for debugging
        print(f"  Detected peak timepoints: {timepoints[peak_indices]}")
        print(f"  Detected peak indices: {peak_indices}")

    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(timepoints, signal, label='Original Signal', color='gray')
        plt.plot(timepoints, baseline, label='Baseline', color='orange')
        plt.plot(timepoints, corrected_signal, label='Corrected Signal', color='blue')
        plt.plot(timepoints[peak_indices], corrected_signal[peak_indices], "x", color='red',
                 label=f'Peaks (Prom ≥ {prominence:.2f})')

        plt.xlabel('Timepoint')
        plt.ylabel('Signal')
        plt.title(f'Peak Detection with Prominence on {column_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    return peak_indices, corrected_signal, baseline, normalized_signal


class Chromatogram:
    """
    Class for handling chromatography data, including baseline correction and peak fitting.
    """
    
    def __init__(self, df, cols={'time': 'Timepoint', 'signal': None}):
        """
        Initialize a Chromatogram object from a DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame containing chromatography data
            cols (dict): Dictionary mapping of column names with keys 'time' and 'signal'
        """
        self.df = df.copy()
        
        # Assign time column
        if cols['time'] in self.df.columns:
            self.time_col = cols['time']
        else:
            raise ValueError(f"Time column '{cols['time']}' not found in DataFrame")
            
        # Assign signal column
        if cols['signal'] is None:
            # If no signal column specified, use first non-time column
            signal_cols = [col for col in self.df.columns if col != self.time_col]
            if not signal_cols:
                raise ValueError("No signal columns found in DataFrame")
            self.signal_col = signal_cols[0]
        elif cols['signal'] in self.df.columns:
            self.signal_col = cols['signal']
        else:
            raise ValueError(f"Signal column '{cols['signal']}' not found in DataFrame")
        
        # Extract the time and signal arrays
        self.time = self.df[self.time_col].values
        self.signal = self.df[self.signal_col].values
        
        # Initialize other attributes
        self.baseline = None
        self.corrected_signal = None
        self.peaks = None
        self.peak_fits = None
        
    def crop(self, time_range):
        """
        Crop the chromatogram to a specific time range.
        
        Args:
            time_range (list): [min_time, max_time] to crop to
        """
        mask = (self.df[self.time_col] >= time_range[0]) & (self.df[self.time_col] <= time_range[1])
        self.df = self.df[mask].copy()
        self.time = self.df[self.time_col].values
        self.signal = self.df[self.signal_col].values
        
        # Reset any processed data
        self.baseline = None
        self.corrected_signal = None
        self.peaks = None
        self.peak_fits = None
        
        return self
    
    def show(self, title=None):
        """
        Plot the chromatogram.
        
        Args:
            title (str, optional): Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot raw signal
        ax.plot(self.time, self.signal, 'k-', label='raw chromatogram')
        
        # If baseline correction has been performed, plot that too
        if self.corrected_signal is not None:
            ax.plot(self.time, self.corrected_signal, 'b-', label='baseline corrected')
            
        # If peaks have been found, mark them
        if self.peaks is not None:
            ax.plot(self.time[self.peaks], self.corrected_signal[self.peaks], 'ro', label='peaks')
            
        # Set labels and title
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(self.signal_col)
        if title:
            ax.set_title(title)
            
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def correct_baseline(self, window_size=50):
        """
        Apply baseline correction using SNIP algorithm.
        
        Args:
            window_size (int): Window size for SNIP algorithm
        """
        self.corrected_signal, self.baseline = snip_baseline(self.signal, window_size)
        return self
    
    def fit_peaks(self, prominence=0.01, width=5, buffer=0.1):
        """
        Find and fit peaks in the chromatogram.
        
        Args:
            prominence (float): Minimum peak prominence
            width (int): Minimum peak width
            buffer (float): Buffer region around peaks
        """
        from tqdm.auto import tqdm
        
        # Perform baseline correction if not already done
        if self.corrected_signal is None:
            print("Performing baseline correction:")
            self.correct_baseline()
        
        # Find peaks with scipy.signal.find_peaks
        peak_indices, properties = scipy.signal.find_peaks(
            self.corrected_signal, 
            prominence=prominence, 
            width=width
        )
        
        self.peaks = peak_indices
        
        # Define Gaussian peak function
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        # Fit each peak with a Gaussian
        self.peak_fits = []
        
        for peak_idx in tqdm(peak_indices, desc="Deconvolving mixture"):
            # Extract region around the peak
            peak_time = self.time[peak_idx]
            
            # Find closest points to the left and right of the peak ± buffer
            left_idx = np.abs(self.time - (peak_time - buffer)).argmin()
            right_idx = np.abs(self.time - (peak_time + buffer)).argmin()
            
            # Ensure we have enough points
            if right_idx - left_idx < 3:
                continue
                
            x_data = self.time[left_idx:right_idx+1]
            y_data = self.corrected_signal[left_idx:right_idx+1]
            
            # Initial guess for Gaussian parameters
            amp_guess = self.corrected_signal[peak_idx]
            mu_guess = peak_time
            sigma_guess = 0.1  # Arbitrary initial guess
            
            try:
                # Fit the Gaussian
                popt, _ = curve_fit(gaussian, x_data, y_data, p0=[amp_guess, mu_guess, sigma_guess])
                
                # Store the parameters
                self.peak_fits.append({
                    'index': peak_idx,
                    'time': self.time[peak_idx],
                    'amp': popt[0],
                    'mu': popt[1],
                    'sigma': popt[2],
                    'area': popt[0] * popt[2] * np.sqrt(2 * np.pi)  # Area of Gaussian
                })
            except:
                # If fitting fails, store with NaN parameters
                self.peak_fits.append({
                    'index': peak_idx,
                    'time': self.time[peak_idx],
                    'amp': np.nan,
                    'mu': np.nan,
                    'sigma': np.nan,
                    'area': np.nan
                })
                
        return self
    
    def assess_fit(self):
        """
        Assess the quality of peak fitting.
        
        Returns:
            pd.DataFrame: Dataframe with fit quality metrics
        """
        if self.peaks is None or self.corrected_signal is None:
            raise ValueError("Must run fit_peaks() before assessing fit")
            
        # Calculate the reconstructed chromatogram
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        reconstructed = np.zeros_like(self.time, dtype=float)
        for peak in self.peak_fits:
            if not np.isnan(peak['amp']):
                reconstructed += gaussian(self.time, peak['amp'], peak['mu'], peak['sigma'])
                
        # Find continuous regions of interest
        # 1. Start with peak regions
        is_peak_region = np.zeros_like(self.time, dtype=bool)
        
        for i, peak in enumerate(self.peak_fits):
            peak_idx = peak['index']
            
            # Find points where signal is >1% of peak height
            threshold = 0.01 * self.corrected_signal[peak_idx]
            
            # Scan left
            left_idx = peak_idx
            while left_idx > 0 and self.corrected_signal[left_idx] > threshold:
                left_idx -= 1
                
            # Scan right
            right_idx = peak_idx
            while right_idx < len(self.time)-1 and self.corrected_signal[right_idx] > threshold:
                right_idx += 1
                
            # Mark this region
            is_peak_region[left_idx:right_idx+1] = True
            
        # 2. Identify interpeak regions
        interpeak_regions = []
        peak_regions = []
        
        # Variables to track regions
        in_region = False
        start_idx = 0
        is_peak = False
        
        for i in range(len(self.time)):
            if not in_region:
                # Starting a new region
                if self.corrected_signal[i] > 0.000001:
                    in_region = True
                    start_idx = i
                    is_peak = is_peak_region[i]
            else:
                # In middle of a region
                if is_peak != is_peak_region[i]:
                    # Region type changed, end this region
                    if is_peak:
                        peak_regions.append((start_idx, i-1))
                    else:
                        interpeak_regions.append((start_idx, i-1))
                    # Start a new region
                    start_idx = i
                    is_peak = is_peak_region[i]
                elif i == len(self.time) - 1:
                    # Last point, end the region
                    if is_peak:
                        peak_regions.append((start_idx, i))
                    else:
                        interpeak_regions.append((start_idx, i))
                    in_region = False
                    
        # Calculate quality metrics for each region
        results = []
        
        # Define a function to calculate R-score (ratio of areas)
        def r_score(signal_area, reconstructed_area):
            if signal_area == 0:
                return np.inf
            return reconstructed_area / signal_area
        
        # Process interpeak regions
        for i, (start_idx, end_idx) in enumerate(interpeak_regions):
            region_time = self.time[start_idx:end_idx+1]
            region_signal = self.corrected_signal[start_idx:end_idx+1]
            region_reconstructed = reconstructed[start_idx:end_idx+1]
            
            # Skip very small regions
            if len(region_time) < 3:
                continue
                
            # Calculate metrics
            signal_area = np.trapz(region_signal, region_time)
            reconstructed_area = np.trapz(region_reconstructed, region_time)
            r = r_score(signal_area, reconstructed_area)
            
            # Calculate variance and Fano factor
            signal_variance = np.var(region_signal)
            signal_mean = np.mean(region_signal)
            fano = signal_variance / (signal_mean + 1e-10)  # Avoid division by zero
            
            results.append({
                'window_id': i+1,
                'time_start': region_time[0],
                'time_end': region_time[-1],
                'signal_area': signal_area,
                'inferred_area': reconstructed_area,
                'signal_variance': signal_variance,
                'signal_mean': signal_mean,
                'signal_fano_factor': fano,
                'reconstruction_score': r,
                'window_type': 'interpeak'
            })
            
        # Process peak regions
        for i, (start_idx, end_idx) in enumerate(peak_regions):
            region_time = self.time[start_idx:end_idx+1]
            region_signal = self.corrected_signal[start_idx:end_idx+1]
            region_reconstructed = reconstructed[start_idx:end_idx+1]
            
            # Skip very small regions
            if len(region_time) < 3:
                continue
                
            # Calculate metrics
            signal_area = np.trapz(region_signal, region_time)
            reconstructed_area = np.trapz(region_reconstructed, region_time)
            r = r_score(signal_area, reconstructed_area)
            
            # Calculate variance and Fano factor
            signal_variance = np.var(region_signal)
            signal_mean = np.mean(region_signal)
            fano = signal_variance / (signal_mean + 1e-10)  # Avoid division by zero
            
            results.append({
                'window_id': i+1,
                'time_start': region_time[0],
                'time_end': region_time[-1],
                'signal_area': signal_area,
                'inferred_area': reconstructed_area,
                'signal_variance': signal_variance,
                'signal_mean': signal_mean,
                'signal_fano_factor': fano,
                'reconstruction_score': r,
                'window_type': 'peak'
            })
            
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Apply tolerance criteria for R-score
        tolerance = 0.01  # 1% tolerance
        
        df_results['applied_tolerance'] = tolerance
        
        # Apply different criteria based on window type
        peak_mask = df_results['window_type'] == 'peak'
        interpeak_mask = df_results['window_type'] == 'interpeak'
        
        # For peak regions, check if R is within 1 ± tolerance
        df_results.loc[peak_mask, 'status'] = np.where(
            (df_results.loc[peak_mask, 'reconstruction_score'] >= 1-tolerance) & 
            (df_results.loc[peak_mask, 'reconstruction_score'] <= 1+tolerance),
            'valid', 'needs review'
        )
        
        # Get max Fano factor from peak regions for comparison
        if peak_mask.any():
            max_peak_fano = df_results.loc[peak_mask, 'signal_fano_factor'].max()
        else:
            max_peak_fano = 1.0
            
        # For interpeak regions, compare Fano factor to peak regions
        df_results.loc[interpeak_mask, 'status'] = np.where(
            (df_results.loc[interpeak_mask, 'signal_fano_factor'] <= 0.001 * max_peak_fano),
            'low signal', 'needs review'
        )
        
        # Generate a report
        print("\n-------------------Chromatogram Reconstruction Report Card----------------------\n")
        
        print("Reconstruction of Peaks")
        print("=======================\n")
        for _, row in df_results[df_results['window_type'] == 'peak'].iterrows():
            if row['status'] == 'valid':
                print(f"\033[1m\033[42m\033[30mA+, Success:  Peak Window {row['window_id']} (t: {row['time_start']:.3f} - {row['time_end']:.3f}) R-Score = {row['reconstruction_score']:.4f}\033[0m\n")
            else:
                print(f"\033[1m\033[41m\033[37mF, Failed:  Peak Window {row['window_id']} (t: {row['time_start']:.3f} - {row['time_end']:.3f}) R-Score = {row['reconstruction_score']:.4f}\033[0m\n")
                print(f"Peak region {row['window_id']} is not well reconstructed by Gaussian mixture.")
                print(f"This indicates that the peak may have a non-Gaussian shape, or that multiple")
                print(f"overlapping peaks might be present that weren't resolved.\n")
        
        print("Signal Reconstruction of Interpeak Windows")
        print("==========================================")
        print("                  ")
        for _, row in df_results[df_results['window_type'] == 'interpeak'].iterrows():
            if row['status'] == 'low signal':
                continue
            elif row['reconstruction_score'] > 1.15:
                print(f"\033[1m\033[43m\033[30mC-, Needs Review:  Interpeak Window {row['window_id']} (t: {row['time_start']:.3f} - {row['time_end']:.3f}) R-Score = {row['reconstruction_score']:.4f} & Fano Ratio = {row['signal_fano_factor']/max_peak_fano:.4f}\033[0m")
                print(f"Interpeak window {row['window_id']} is not well reconstructed by mixture, but has a small Fano factor")
                print(f"compared to peak region(s). This is likely acceptable, but visually check this region.\n")
            else:
                print(f"\033[1m\033[42m\033[30mA, Good:  Interpeak Window {row['window_id']} (t: {row['time_start']:.3f} - {row['time_end']:.3f}) R-Score = {row['reconstruction_score']:.4f}\033[0m\n")
        
        print("\n--------------------------------------------------------------------------------\n")
        
        return df_results


def analyze_chromatogram(combined_df, column_name, prominence=0.5, width=20,
                       shoulder_curvature=0.005, max_peaks_per_window=3, plot=True):
    """Analyzes a chromatogram column."""
    if 'Timepoint' not in combined_df.columns or column_name not in combined_df.columns:
        print("Error: Required column(s) not found.")
        return None

    signal = combined_df[column_name].values
    timepoints = combined_df['Timepoint'].values

    # --- Step 1: Baseline Correction and Normalization ---
    corrected_signal, baseline, sg_window_size = normalize_signal(signal, timepoints)
    
    # --- Step 2: Find Main Peaks ---
    peak_indices, normalized_corrected_signal = find_main_peaks(corrected_signal, timepoints, prominence, width, sg_window_size)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(timepoints, normalized_corrected_signal, label='Corrected Signal', color='blue')
        plt.plot(timepoints[peak_indices], normalized_corrected_signal[peak_indices], "x", color='red', label='Peaks')
        plt.title('Step 2: Main Peak Detection')
        plt.xlabel('Timepoint')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Step 3: Detect Shoulders ---
    shoulder_indices = detect_shoulders(normalized_corrected_signal, timepoints, peak_indices,
                                       sg_window_size, baseline, shoulder_curvature, width)

    if plot:  # Plot after Step 3
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(timepoints, signal, label='Original Signal', color='gray') #plot original
        ax1.plot(timepoints, corrected_signal, label='Corrected Signal', color='blue')
        ax1.plot(timepoints, baseline, label='Baseline', color='orange')
        ax1.plot(timepoints[peak_indices], corrected_signal[peak_indices], "x", color='red', label='Peaks')
        ax1.plot(timepoints[shoulder_indices], corrected_signal[shoulder_indices], "o",
                color='green', label='Shoulders', markersize=5)
        ax1.set_title('Peak and Shoulder Detection')
        ax1.set_ylabel('Signal')
        ax1.legend()
        ax1.grid(True)

        # Plot second derivative in the second subplot
        second_derivative = scipy.signal.savgol_filter(normalized_corrected_signal, window_length=sg_window_size, polyorder=2, deriv=2)
        ax2.plot(timepoints, -second_derivative * 10, label='-2nd Deriv (Scaled)', color='purple')  # Scaled for visibility
        ax2.set_xlabel('Timepoint')
        ax2.set_ylabel('Second Derivative (Scaled)')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    # --- Step 4: Return Results ---
    results = {
        'column_name': column_name,
        'peak_indices': peak_indices,
        'shoulder_indices': shoulder_indices,
        'corrected_signal': normalized_corrected_signal,
        'baseline': baseline,
        'sg_window_size': sg_window_size
    }
    return results
# fseccnn/emg_fitter.py

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import exponnorm
from scipy.integrate import simpson
import matplotlib.pyplot as plt  # For standalone plotting method
import logging
import sys  # For standalone test exit

# --- Setup logging ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())


# --- EMG Function ---
def emg_peak_scipy(x, amplitude_scaler, mu, sigma, tau):
    sigma = max(sigma, 1e-7)
    tau = max(tau, 1e-7)
    K = tau / sigma
    pdf_values = exponnorm.pdf(x, K=K, loc=mu - tau, scale=sigma)
    return amplitude_scaler * pdf_values


# --- Sum of EMGs Model Function (for curve_fit) ---
def sum_of_emgs_model(x, *params):
    """
    Calculates the sum of k EMG peaks + constant baseline offset.
    params: A flat list:
            [A1, mu1, sigma1, tau1, ..., Ak, muk, sigmak, tauk, baseline_offset]
    """
    num_peak_params = 4  # A_scaler, mu, sigma, tau
    num_baseline_params = 1  # Only offset

    if (len(params) - num_baseline_params) % num_peak_params != 0:
        raise ValueError(
            f"Number of parameters ({len(params)}) is not consistent with "
            f"{num_peak_params} params per peak and {num_baseline_params} baseline param(s)."
        )
    num_peaks = (len(params) - num_baseline_params) // num_peak_params
    y_fit = np.zeros_like(x, dtype=float)

    for i in range(num_peaks):
        start_idx = i * num_peak_params
        A_i, mu_i, sigma_i, tau_i = params[start_idx: start_idx + num_peak_params]
        y_fit += emg_peak_scipy(x, A_i, mu_i, sigma_i, tau_i)

    baseline_offset = params[-num_baseline_params]  # Get the single baseline offset
    y_fit += baseline_offset  # Add constant offset
    return y_fit


class EMGFitter:
    def __init__(self, x_data, y_data, detected_peak_indices, config=None):
        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
            raise TypeError("x_data and y_data must be numpy arrays.")
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length.")

        self.num_peak_params_per_component = 4
        self.num_baseline_params = 1  # UPDATED: Only constant offset

        if len(x_data) < 2:
            logger.warning(
                "EMGFitter initialized with less than 2 data points. Fitting will likely fail or be trivial.")
            self.x_data = x_data.copy() if isinstance(x_data, np.ndarray) else np.array(x_data)
            self.y_data = y_data.copy() if isinstance(y_data, np.ndarray) else np.array(y_data)
            self.detected_peak_indices = []
            self.num_peaks = 0  # Number of peaks to attempt fitting
        else:
            self.x_data = x_data.copy()
            self.y_data = y_data.copy()

            if isinstance(detected_peak_indices, np.ndarray):
                initial_indices = detected_peak_indices.tolist()
            elif isinstance(detected_peak_indices, list):
                initial_indices = list(detected_peak_indices)
            else:
                logger.warning(
                    f"detected_peak_indices is of unexpected type {type(detected_peak_indices)}. Attempting conversion.")
                try:
                    initial_indices = list(detected_peak_indices)
                except TypeError:
                    logger.error("Could not convert detected_peak_indices to a list. No peaks will be fitted.")
                    initial_indices = []

            valid_indices = sorted(list(set(int(idx) for idx in initial_indices if 0 <= int(idx) < len(self.x_data))))
            if len(valid_indices) != len(initial_indices):
                logger.info(
                    f"Filtered {len(initial_indices) - len(valid_indices)} initial peak indices due to invalid values or out of bounds.")
            self.detected_peak_indices_cnn = valid_indices

        self.default_config = {
            "sigma_guess_fraction": 0.05,
            "tau_guess_factor": 0.1,
            "amplitude_bound_factor": 2.0,
            "sigma_bound_max_fraction": 0.3,
            "mu_bound_padding_factor": 0.05,
            "tau_bound_max_fraction": 0.5,
            "max_nfev": None,
            "fit_method": 'trf',
            "min_signal_height_for_fit": 0.0,
            "min_signal_prominence_for_fit": 0.0,
            "remove_peaks_with_sigma_too_small_factor": 2.0,
            "remove_peaks_with_sigma_too_large_factor": 0.5
        }
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        self._apply_pre_fit_filters()  # Updates self.detected_peak_indices and self.num_peaks

        self.fitted_params_all = None
        self.pcov = None
        self.fit_successful = False

        self.fitted_params_list = []
        self.individual_peak_shapes = []
        self.fitted_baseline_shape = np.zeros_like(self.x_data) if len(self.x_data) > 0 else np.array([])

        if self.config["max_nfev"] is None and self.num_peaks > 0:
            self.config["max_nfev"] = 300 * self.num_peaks * self.num_peak_params_per_component
        elif self.config["max_nfev"] is None and self.num_peaks == 0:
            self.config["max_nfev"] = 1000

    def _apply_pre_fit_filters(self):
        if not hasattr(self, 'detected_peak_indices_cnn') or not self.detected_peak_indices_cnn:
            self.detected_peak_indices = []
            self.num_peaks = 0
            return

        indices_to_process = list(self.detected_peak_indices_cnn)

        min_height = self.config.get("min_signal_height_for_fit", 0.0)
        if min_height > 0.0 and indices_to_process:
            passed_height_filter = [
                idx for idx in indices_to_process if self.y_data[idx] >= min_height
            ]
            if len(passed_height_filter) < len(indices_to_process):
                logger.info(
                    f"EMGFitter: Filtered {len(indices_to_process) - len(passed_height_filter)} peaks (of {len(indices_to_process)}) by min_signal_height_for_fit ({min_height:.3f}).")
            indices_to_process = passed_height_filter

        min_prominence = self.config.get("min_signal_prominence_for_fit", 0.0)
        if min_prominence > 0.0 and indices_to_process:
            try:
                from scipy.signal import peak_prominences
                if indices_to_process:
                    proms, _, _ = peak_prominences(self.y_data, indices_to_process)
                    passed_prominence_filter = [
                        idx for i, idx in enumerate(indices_to_process) if proms[i] >= min_prominence
                    ]
                    if len(passed_prominence_filter) < len(indices_to_process):
                        logger.info(
                            f"EMGFitter: Filtered {len(indices_to_process) - len(passed_prominence_filter)} peaks by min_signal_prominence_for_fit ({min_prominence:.3f}).")
                    indices_to_process = passed_prominence_filter
                else:
                    logger.info("EMGFitter: No peaks left to filter by prominence.")
            except ImportError:
                logger.warning("scipy.signal.peak_prominences not available. Skipping prominence filter.")
            except Exception as e:
                logger.warning(f"Error during prominence calculation: {e}. Skipping prominence filter.")

        self.detected_peak_indices = sorted(list(set(indices_to_process)))
        self.num_peaks = len(self.detected_peak_indices)
        logger.info(f"EMGFitter: Proceeding to fit {self.num_peaks} peaks after pre-fit filters.")

    def _prepare_initial_guesses_and_bounds(self):
        if self.num_peaks == 0:  # For baseline-only fit
            p0_baseline = [np.min(self.y_data) if len(self.y_data) > 0 else 0]  # offset only
            bounds_baseline_low = [np.min(self.y_data) - np.ptp(self.y_data) if len(self.y_data) > 1 else -1]
            bounds_baseline_high = [np.max(self.y_data) + np.ptp(self.y_data) if len(self.y_data) > 1 else 1]
            return p0_baseline, (bounds_baseline_low, bounds_baseline_high)

        p0_flat, bounds_l_flat, bounds_u_flat = [], [], []

        x_min_data, x_max_data = self.x_data[0], self.x_data[-1]
        x_range_data = x_max_data - x_min_data if x_max_data > x_min_data else 1.0
        y_min_data, y_max_data = np.min(self.y_data), np.max(self.y_data)
        y_ptp_data = y_max_data - y_min_data if y_max_data > y_min_data else 1.0

        min_x_step = np.min(np.diff(self.x_data)) if len(np.diff(self.x_data)) > 0 else 0.01 * x_range_data
        min_x_step = max(min_x_step, 1e-6)

        # Initial Baseline Offset Estimate (can be refined)
        # A simple estimate: median of the lowest 20% of y_data values, or just min y_data
        if len(self.y_data) > 5:
            init_baseline_offset = np.median(np.sort(self.y_data)[:max(1, len(self.y_data) // 5)])
        else:
            init_baseline_offset = y_min_data if len(self.y_data) > 0 else 0.0

        for peak_idx_in_original_data in self.detected_peak_indices:
            mu_g = self.x_data[peak_idx_in_original_data]
            mu_bound_shift = x_range_data * self.config.get("mu_bound_padding_factor", 0.05)
            mu_l = max(x_min_data, mu_g - mu_bound_shift)
            mu_u = min(x_max_data, mu_g + mu_bound_shift)
            if mu_l >= mu_u: mu_l = x_min_data; mu_u = x_max_data
            mu_g = np.clip(mu_g, mu_l, mu_u)

            sigma_l = min_x_step * 1.5
            sigma_u = x_range_data * self.config["sigma_bound_max_fraction"]
            sigma_u = max(sigma_u, sigma_l * 2)
            sigma_g = x_range_data * self.config["sigma_guess_fraction"] / np.sqrt(max(1, self.num_peaks))
            sigma_g = np.clip(sigma_g, sigma_l, sigma_u)

            y_at_peak_eff = self.y_data[peak_idx_in_original_data] - init_baseline_offset  # Subtract estimated offset
            y_at_peak_eff = max(y_at_peak_eff, 0.01 * y_ptp_data)
            A_scaler_g = y_at_peak_eff * sigma_g
            A_scaler_l = 1e-7 * y_ptp_data * min_x_step
            A_scaler_u = y_ptp_data * x_range_data * self.config["amplitude_bound_factor"]
            A_scaler_g = np.clip(A_scaler_g, A_scaler_l, A_scaler_u)

            tau_l = 1e-7
            tau_u = x_range_data * self.config["tau_bound_max_fraction"]
            tau_u = max(tau_u, tau_l * 10)
            tau_g = sigma_g * self.config["tau_guess_factor"]
            tau_g = np.clip(tau_g, tau_l, tau_u)

            p0_flat.extend([A_scaler_g, mu_g, sigma_g, tau_g])
            bounds_l_flat.extend([A_scaler_l, mu_l, sigma_l, tau_l])
            bounds_u_flat.extend([A_scaler_u, mu_u, sigma_u, tau_u])

        # Baseline Offset Parameter
        # Allow baseline to go below min data point and above max, but center guess
        bl_offset_l = y_min_data - 0.5 * y_ptp_data  # More constrained than linear version
        bl_offset_u = y_max_data + 0.2 * y_ptp_data
        # Refine init_baseline_offset if all peaks are well above current estimate
        if self.num_peaks > 0:
            min_peak_y_guess = np.min([p0_flat[i * self.num_peak_params_per_component] for i in range(self.num_peaks)])
            # This is A_scaler, not y_at_peak directly. Simpler estimate of overall signal floor.
            # A better estimate for init_baseline_offset if peaks are present:
            lowest_signal_regions = np.sort(self.y_data)[:max(5, len(self.y_data) // 10)]
            init_baseline_offset = np.median(lowest_signal_regions)

        init_baseline_offset = np.clip(init_baseline_offset, bl_offset_l, bl_offset_u)

        p0_flat.extend([init_baseline_offset])
        bounds_l_flat.extend([bl_offset_l])
        bounds_u_flat.extend([bl_offset_u])

        for i in range(len(p0_flat)):
            if bounds_l_flat[i] >= bounds_u_flat[i]:
                logger.warning(
                    f"Correcting bound at index {i}: low ({bounds_l_flat[i]:.3e}) >= high ({bounds_u_flat[i]:.3e}).")
                bounds_u_flat[i] = bounds_l_flat[i] + abs(bounds_l_flat[i] * 0.1) + 1e-3
                if bounds_l_flat[i] >= bounds_u_flat[i]: bounds_u_flat[i] = bounds_l_flat[i] * 1.1 if bounds_l_flat[
                                                                                                          i] > 1e-9 else 1e-3
            p0_flat[i] = np.clip(p0_flat[i], bounds_l_flat[i], bounds_u_flat[i])
            if bounds_l_flat[i] >= bounds_u_flat[i]:
                logger.error(
                    f"FATAL BOUND ERROR POST-CORRECTION at index {i}: low ({bounds_l_flat[i]:.3e}) >= high ({bounds_u_flat[i]:.3e}). P0={p0_flat[i]:.3e}")
        return p0_flat, (bounds_l_flat, bounds_u_flat)

    def fit(self):
        if self.num_peaks == 0 and len(self.x_data) > 1:
            logger.info("No peaks to fit after pre-filters. Attempting to fit constant baseline only.")
            p0_baseline, bounds_baseline = self._prepare_initial_guesses_and_bounds()
            try:
                def baseline_only_model(x, offset):  # Only offset
                    return np.full_like(x, offset)  # Return array of same shape as x

                self.fitted_params_all, self.pcov = curve_fit(
                    baseline_only_model, self.x_data, self.y_data,
                    p0=p0_baseline, bounds=bounds_baseline,
                    maxfev=self.config.get("max_nfev", 1000), method=self.config["fit_method"]
                )
                self.fitted_baseline_shape = baseline_only_model(self.x_data, *self.fitted_params_all)
                self.fit_successful = True
                logger.info(f"Constant baseline-only fit successful. Offset: {self.fitted_params_all[0]:.3f}")
            except Exception as e:
                logger.error(f"Constant baseline-only fitting failed: {e}")
                self.fit_successful = False
                self.fitted_baseline_shape = np.zeros_like(self.x_data)
            return self.fit_successful

        elif self.num_peaks == 0 and len(self.x_data) <= 1:
            logger.warning("No peaks and insufficient data for baseline fitting. Fit skipped.")
            self.fit_successful = False
            return False

        p0, bounds = self._prepare_initial_guesses_and_bounds()
        logger.debug(f"Fitting {self.num_peaks} peaks. Initial p0: {np.round(p0, 3)}")
        logger.debug(f"Bounds L: {np.round(bounds[0], 3)}, U: {np.round(bounds[1], 3)}")

        try:
            self.fitted_params_all, self.pcov = curve_fit(
                sum_of_emgs_model, self.x_data, self.y_data,
                p0=p0, bounds=bounds,
                maxfev=self.config["max_nfev"], method=self.config["fit_method"],
            )
            self.fit_successful = True
            self._parse_and_store_fitted_components()
            self._apply_post_fit_filters()
            logger.info(f"EMG fitting successful. Found {len(self.fitted_params_list)} components after post-filters.")
        except RuntimeError as e:
            logger.error(f"RuntimeError during fitting: {e}. Fit failed. Using initial guesses for components.")
            self.fitted_params_all = np.array(p0)
            self.pcov = np.full((len(p0), len(p0)), np.nan)
            self.fit_successful = False
            self._parse_and_store_fitted_components(use_initial_guesses_if_failed=True)
        except ValueError as e:
            logger.error(f"ValueError during fitting (bounds/p0): {e}. Fit failed. Using initial guesses.")
            self.fitted_params_all = np.array(p0)
            self.pcov = np.full((len(p0), len(p0)), np.nan)
            self.fit_successful = False
            self._parse_and_store_fitted_components(use_initial_guesses_if_failed=True)
        return self.fit_successful

    def _parse_and_store_fitted_components(self, use_initial_guesses_if_failed=False):
        params_to_use = self.fitted_params_all
        if use_initial_guesses_if_failed and (self.fitted_params_all is None or not self.fit_successful):
            p0_fallback, _ = self._prepare_initial_guesses_and_bounds()
            if len(p0_fallback) == (self.num_peaks * self.num_peak_params_per_component + self.num_baseline_params):
                params_to_use = np.array(p0_fallback)
                logger.info("Using initial guesses (p0) to parse components due to fit failure.")
            else:
                logger.error("Cannot parse components: p0 shape mismatch or fit params unavailable.")
                self.fitted_params_list = []
                self.individual_peak_shapes = []
                self.fitted_baseline_shape = np.zeros_like(self.x_data) if len(self.x_data) > 0 else np.array([])
                return

        if params_to_use is None or len(params_to_use) < self.num_baseline_params:
            logger.error("Cannot parse components: fitted_params_all is None or too short.")
            self.fitted_params_list = []
            self.individual_peak_shapes = []
            self.fitted_baseline_shape = np.zeros_like(self.x_data) if len(self.x_data) > 0 else np.array([])
            return

        self.fitted_params_list = []
        self.individual_peak_shapes = []

        fitted_baseline_offset = params_to_use[-self.num_baseline_params]  # Single offset
        self.fitted_baseline_shape = np.full_like(self.x_data, fitted_baseline_offset)  # Constant baseline

        num_peaks_from_params = (len(params_to_use) - self.num_baseline_params) // self.num_peak_params_per_component

        for i in range(num_peaks_from_params):
            start_idx = i * self.num_peak_params_per_component
            A, mu, sigma, tau = params_to_use[start_idx: start_idx + self.num_peak_params_per_component]
            sigma = max(sigma, 1e-7)
            tau = max(tau, 1e-7)
            self.fitted_params_list.append({'A_scaler': A, 'mu': mu, 'sigma': sigma, 'tau': tau,
                                            'original_cnn_idx_pos': i})
            peak_i_component_shape = emg_peak_scipy(self.x_data, A, mu, sigma, tau)
            self.individual_peak_shapes.append(peak_i_component_shape)

    def _apply_post_fit_filters(self):
        if not self.fitted_params_list:
            return
        filtered_params_list = []
        filtered_peak_shapes = []
        min_x_step = np.min(np.diff(self.x_data)) if len(np.diff(self.x_data)) > 0 else 0.01
        x_range_data = self.x_data[-1] - self.x_data[0] if len(self.x_data) > 1 else 1.0
        sigma_too_small_thresh = min_x_step * self.config.get("remove_peaks_with_sigma_too_small_factor", 2.0)
        sigma_too_large_thresh = x_range_data * self.config.get("remove_peaks_with_sigma_too_large_factor", 0.5)

        for i, params_dict in enumerate(self.fitted_params_list):
            sigma = params_dict['sigma']
            passes = True
            reason = ""
            if sigma < sigma_too_small_thresh:
                passes = False
                reason = f"sigma {sigma:.3e} < {sigma_too_small_thresh:.3e} (too small)"
            elif sigma > sigma_too_large_thresh:
                passes = False
                reason = f"sigma {sigma:.3e} > {sigma_too_large_thresh:.3e} (too large)"

            if passes:
                filtered_params_list.append(params_dict)
                if i < len(self.individual_peak_shapes):
                    filtered_peak_shapes.append(self.individual_peak_shapes[i])
            else:
                logger.info(f"Post-fit filter: Removed peak {i + 1} (μ={params_dict['mu']:.2f}) because {reason}.")

        if len(filtered_params_list) < len(self.fitted_params_list):
            logger.info(
                f"Post-fit filters reduced component count from {len(self.fitted_params_list)} to {len(filtered_params_list)}.")
        self.fitted_params_list = filtered_params_list
        self.individual_peak_shapes = filtered_peak_shapes

    def get_fitted_parameters_list(self):
        return self.fitted_params_list

    def get_individual_peak_shapes(self):
        return self.individual_peak_shapes

    def get_fitted_baseline_shape(self):
        return self.fitted_baseline_shape

    def get_total_fit_curve(self):
        if not self.fitted_params_list and not np.any(self.fitted_baseline_shape):
            if self.fitted_params_all is None and self.num_peaks > 0:
                p0, _ = self._prepare_initial_guesses_and_bounds()
                if len(p0) > self.num_baseline_params:
                    return sum_of_emgs_model(self.x_data, *p0)
                elif len(p0) == self.num_baseline_params:
                    return np.full_like(self.x_data, p0[0])  # Constant baseline
            return np.zeros_like(self.x_data) if len(self.x_data) > 0 else np.array([])

        total_fit = np.sum(self.individual_peak_shapes, axis=0) if self.individual_peak_shapes else np.zeros_like(
            self.x_data)
        total_fit += self.fitted_baseline_shape
        return total_fit

    def calculate_peak_metrics(self, peak_index_in_list):
        if not self.fitted_params_list or peak_index_in_list >= len(self.fitted_params_list) or \
                not self.individual_peak_shapes or peak_index_in_list >= len(self.individual_peak_shapes):
            logger.warning(f"Metrics requested for invalid peak index {peak_index_in_list} or missing data.")
            return None

        params = self.fitted_params_list[peak_index_in_list]
        peak_shape_no_baseline = self.individual_peak_shapes[peak_index_in_list]
        auc = simpson(peak_shape_no_baseline, self.x_data) if len(self.x_data) > 1 else 0.0

        if len(peak_shape_no_baseline) > 0:
            peak_max_idx = np.argmax(peak_shape_no_baseline)
            peak_actual_height_on_baseline = peak_shape_no_baseline[peak_max_idx]
            peak_actual_height_total = peak_actual_height_on_baseline + self.fitted_baseline_shape[peak_max_idx]
            peak_max_x_pos = self.x_data[peak_max_idx]
        else:
            peak_actual_height_on_baseline = np.nan
            peak_actual_height_total = np.nan
            peak_max_x_pos = np.nan

        fwhm = np.nan
        if len(peak_shape_no_baseline) > 1 and peak_actual_height_on_baseline > 1e-9:
            try:
                half_max = peak_actual_height_on_baseline / 2.0
                above_hm_indices = np.where(peak_shape_no_baseline >= half_max)[0]
                if len(above_hm_indices) > 0:
                    left_idx = above_hm_indices[0]
                    if left_idx > 0 and peak_shape_no_baseline[left_idx] > half_max:
                        x1, y1 = self.x_data[left_idx - 1], peak_shape_no_baseline[left_idx - 1]
                        x2, y2 = self.x_data[left_idx], peak_shape_no_baseline[left_idx]
                        x_left_hm = x1 + (x2 - x1) * (half_max - y1) / (y2 - y1) if (y2 - y1) != 0 else x2
                    else:
                        x_left_hm = self.x_data[left_idx]
                    right_idx = above_hm_indices[-1]
                    if right_idx < len(self.x_data) - 1 and peak_shape_no_baseline[right_idx] > half_max:
                        x1, y1 = self.x_data[right_idx], peak_shape_no_baseline[right_idx]
                        x2, y2 = self.x_data[right_idx + 1], peak_shape_no_baseline[right_idx + 1]
                        x_right_hm = x1 + (x2 - x1) * (half_max - y1) / (y2 - y1) if (y2 - y1) != 0 else x1
                    else:
                        x_right_hm = self.x_data[right_idx]
                    fwhm = abs(x_right_hm - x_left_hm)
                else:
                    fwhm = 0.0
            except Exception as e:
                logger.debug(f"FWHM calculation error for peak {peak_index_in_list}: {e}")
                fwhm = np.nan

        metrics = {
            "fit_auc": auc,
            "fit_height": peak_actual_height_on_baseline,
            "fit_height_total": peak_actual_height_total,
            "fit_max_pos_x": peak_max_x_pos,
            "fit_fwhm": fwhm,
            "fit_mu": params['mu'],
            "fit_sigma": params['sigma'],
            "fit_tau": params['tau'],
            "fit_amp_scaler": params['A_scaler']
        }
        all_aucs = [simpson(comp_shape, self.x_data) if len(self.x_data) > 1 else 0 for comp_shape in
                    self.individual_peak_shapes]
        total_fitted_auc = sum(all_aucs)
        if total_fitted_auc > 1e-9:
            metrics["fit_auc_percent"] = (auc / total_fitted_auc) * 100.0
        else:
            metrics["fit_auc_percent"] = 0.0 if len(all_aucs) > 0 else np.nan
        return metrics

    def plot_fit_results(self, title="FSEC Deconvolution Fit", true_params_list=None,
                         show_individual_components_line=True,
                         show_filled_components=True):
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        ax.plot(self.x_data, self.y_data, label="Original Data", color='black', alpha=0.7, linewidth=1.5, zorder=2)
        total_fit_curve = self.get_total_fit_curve()
        baseline_curve = self.get_fitted_baseline_shape()
        individual_final_shapes = self.get_individual_peak_shapes()

        ax.plot(self.x_data, total_fit_curve, label="Total Fit Curve", color='red', linestyle='--', linewidth=1.8,
                zorder=3)
        ax.plot(self.x_data, baseline_curve, label="Fitted Baseline (Constant Offset)", color='purple', linestyle=':',
                alpha=0.9,  # Updated label
                zorder=1)

        if individual_final_shapes:
            num_plotted_peaks = len(individual_final_shapes)
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, num_plotted_peaks)))
            for i in range(num_plotted_peaks):
                component_plus_baseline = individual_final_shapes[i] + baseline_curve
                label_comp = f"Fit Comp. {i + 1}"
                if show_filled_components:
                    ax.fill_between(self.x_data, baseline_curve, component_plus_baseline,
                                    color=colors[i], alpha=0.4,
                                    label=label_comp if not show_individual_components_line else None)
                if show_individual_components_line:
                    ax.plot(self.x_data, component_plus_baseline, color=colors[i], linestyle='-',
                            alpha=0.8, linewidth=1.2, label=label_comp if not show_filled_components else None)

        if true_params_list:
            colors_true = plt.cm.autumn(np.linspace(0.1, 0.9, len(true_params_list)))
            for i, p_true in enumerate(true_params_list):
                tau_true = p_true.get('tau', 1e-7)
                true_comp_shape = emg_peak_scipy(self.x_data, p_true['A'], p_true['mu'], p_true['sigma'], tau_true)
                true_baseline_offset_for_plot = 0  # Assuming true components are also on a zero baseline for comparison
                if 'baseline_offset' in p_true:  # If synthetic data generation includes true baseline offset
                    true_baseline_offset_for_plot = p_true['baseline_offset']

                ax.plot(self.x_data, true_comp_shape + true_baseline_offset_for_plot,
                        color=colors_true[i], linestyle=':', alpha=1.0, linewidth=1.5,
                        label=f"True Comp. {i + 1} {'(S)' if p_true.get('is_shoulder') else ''}", zorder=0)

        ax.set_title(title + (f" (Fit Success: {self.fit_successful})"))
        ax.set_xlabel("Elution Volume / Time");
        ax.set_ylabel("Intensity")
        # Adjust y_lim if data and fit are present
        if len(self.y_data) > 0 and len(total_fit_curve) > 0:
            y_min_plot = min(np.min(self.y_data), np.min(total_fit_curve)) - 0.1 * np.ptp(self.y_data)
            y_max_plot = max(np.max(self.y_data), np.max(total_fit_curve)) + 0.1 * np.ptp(self.y_data)
            ax.set_ylim(y_min_plot, y_max_plot)
        elif len(self.y_data) > 0:
            ax.set_ylim(np.min(self.y_data) - 0.1 * np.ptp(self.y_data),
                        np.max(self.y_data) + 0.1 * np.ptp(self.y_data))

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:  # Ensure there are items to legend
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize='small', loc='best', ncol=max(1, len(by_label) // 5))
        ax.grid(True, linestyle=':', alpha=0.4);
        plt.tight_layout();
        plt.show()


# --- Example Usage (for standalone testing) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Standalone EMGFitter Test (Constant Offset Baseline) ---")

    try:
        from synthetic_data import generate_synthetic_chromatogram_mixed_smooth_noise
    except ImportError:
        logger.error("Could not import synthetic_data. Ensure it's in path for this test.")
        sys.exit(1)

    # Update generator config to use only constant baseline offset for true data generation
    config_gen = {
        "length": 1024, "x_range": (5, 55), "max_peaks": 3, "min_peaks": 2,
        "amplitude_range": (0.5, 1.0), "std_dev_range": (0.8, 1.5), "tau_range": (0.1, 0.8),
        "peak_type_ratio": 0.6, "shoulder_peak_probability": 0.4,
        "shoulder_amplitude_ratio_range": (0.2, 0.4), "shoulder_std_dev_ratio_range": (0.7, 1.0),
        "shoulder_mu_offset_ratio_range": (-0.8, 0.8),
        "base_noise_level": 0.01,
        'baseline_linear_drift_variation': 0.0,  # NO SLOPE for true data
        'baseline_wobble_amplitude_factor': 0.0,  # NO WOBBLE for simpler true data test
        'baseline_y_corrected_min_level': 0.05,  # True data has some positive offset
        'return_true_baseline_params': True  # To get the true offset for comparison
    }
    y_signal, _, true_peak_params_list, x_signal, true_baseline_params = \
        generate_synthetic_chromatogram_mixed_smooth_noise(**config_gen)

    true_baseline_offset_value = 0
    if true_baseline_params and 'baseline_offset' in true_baseline_params:
        true_baseline_offset_value = true_baseline_params['baseline_offset']
        # Add true baseline to true peak params for plotting comparison
        for p in true_peak_params_list:
            p['baseline_offset'] = true_baseline_offset_value

    sim_detected_indices = []
    if true_peak_params_list:
        for p in true_peak_params_list:
            if not p.get('is_shoulder', False) or p['A'] > 0.1:
                sim_detected_indices.append(np.argmin(np.abs(x_signal - p['mu'])))
    sim_detected_indices = sorted(list(set(sim_detected_indices)))
    if not sim_detected_indices and len(x_signal) > 10: sim_detected_indices = [len(x_signal) // 3,
                                                                                2 * len(x_signal) // 3]

    logger.info(f"Simulated CNN detected indices: {sim_detected_indices}")
    logger.info(f"True baseline offset for generated data: {true_baseline_offset_value:.3f}")

    fitter_config = {
        "min_signal_height_for_fit": 0.02,
        "remove_peaks_with_sigma_too_small_factor": 1.5,
        "remove_peaks_with_sigma_too_large_factor": 0.4,
        "max_nfev": 50000
    }
    fitter = EMGFitter(x_signal, y_signal, sim_detected_indices, config=fitter_config)
    success = fitter.fit()
    logger.info(f"Fitting successful: {success}")

    final_fitted_params = fitter.get_fitted_parameters_list()
    logger.info(f"\nFinal Fitted Peak Parameters ({len(final_fitted_params)} components):")

    if fitter.fitted_params_all is not None and len(fitter.fitted_params_all) > 0:
        fitted_offset = fitter.fitted_params_all[-1]  # Last param is the offset
        logger.info(f"Fitted Baseline Offset: {fitted_offset:.4f}")
    else:
        logger.info("No fitted baseline offset available (fit might have failed early).")

    for i, p_fit in enumerate(final_fitted_params):
        logger.info(
            f"  Peak {i + 1}: A_sc={p_fit['A_scaler']:.3f}, μ={p_fit['mu']:.2f}, σ={p_fit['sigma']:.3f}, τ={p_fit['tau']:.4f}")
        metrics = fitter.calculate_peak_metrics(i)
        if metrics:
            logger.info(
                f"    Metrics: AUC={metrics['fit_auc']:.2f} ({metrics.get('fit_auc_percent', 0):.1f}%), Height={metrics['fit_height']:.2f} @ X={metrics['fit_max_pos_x']:.2f}, FWHM={metrics['fit_fwhm']:.2f}")

    if len(x_signal) > 1:
        fitter.plot_fit_results(title="EMGFitter Test - Constant Offset Baseline",
                                true_params_list=true_peak_params_list)
    else:
        logger.warning("Not enough data to plot fit results.")
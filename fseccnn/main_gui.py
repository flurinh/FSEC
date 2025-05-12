# fseccnn/main_gui.py
import sys
import os
import time
import numpy as np
import pandas as pd  # Keep for peeking at CSV header, though utils handles main load
import torch
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QLineEdit, QCheckBox,
    QTextEdit, QSizePolicy, QProgressBar, QMessageBox, QGridLayout, QGroupBox,
    QStatusBar, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor  # For table color coding (if that feature is kept/added later)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt  # For plt.Figure in MplCanvas and colormaps

# --- Import custom project modules ---
try:
    from .model import UNet1D
    from .emg_fitter import EMGFitter
    from . import utils
    from . import visualizations
except ImportError:  # Fallback for running as script
    try:
        from model import UNet1D
        from emg_fitter import EMGFitter
        import utils
        import visualizations
    except ImportError as e_script:
        # Use a basic print here if utils.logging itself failed to import/init
        print(f"CRITICAL ERROR: Could not import custom modules: {e_script}")
        # Attempt to use utils.logging if it was imported partially
        if 'utils' in sys.modules and hasattr(utils, 'logging'):
            utils.logging.critical(f"Failed to import all custom modules: {e_script}", exc_info=True)
        sys.exit(1)


# --- Helper functions specific to GUI model loading (Consider moving to model_handler.py) ---
def get_norm_activation_layers_gui(config_model_params):
    from torch import nn
    norm_layer_map = {"BatchNorm1d": nn.BatchNorm1d, "InstanceNorm1d": nn.InstanceNorm1d, "None": None}
    activation_fn_map = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU}
    output_activation_map = {"Sigmoid": nn.Sigmoid, "None": None}

    norm_layer_str = config_model_params.get("norm_layer_str", "BatchNorm1d")
    activation_fn_str = config_model_params.get("activation_fn_str", "ReLU")
    output_activation_str = config_model_params.get("output_activation_str", "Sigmoid")

    norm_layer = norm_layer_map.get(norm_layer_str)
    activation_fn_cls = activation_fn_map.get(activation_fn_str)
    output_activation_cls = output_activation_map.get(output_activation_str)

    if norm_layer_str != "None" and norm_layer is None:
        utils.logging.warning(f"Unknown norm_layer_str: {norm_layer_str}. Defaulting to None.")
        norm_layer = None
    if activation_fn_cls is None:
        utils.logging.warning(f"Unknown activation_fn_str: {activation_fn_str}. Defaulting to ReLU.")
        activation_fn_cls = nn.ReLU
    if output_activation_str != "None" and output_activation_cls is None:  # e.g. if model outputs logits
        utils.logging.info(
            f"Output_activation_str is '{output_activation_str}', model expected to output logits or apply activation internally if not 'None'.")
        # If "None", it implies model outputs logits, and sigmoid will be applied later if needed.
        # If it's an unknown string, then it's an issue.
        if output_activation_str not in output_activation_map:
            utils.logging.warning(
                f"Unknown output_activation_str: {output_activation_str}. Defaulting to None (logits assumed).")
            output_activation_cls = None

    return norm_layer, activation_fn_cls, output_activation_cls


def load_model_for_gui(model_path, train_config_model_params, device):
    norm_layer_cls, activation_fn_cls, output_activation_cls = get_norm_activation_layers_gui(train_config_model_params)
    model = UNet1D(
        in_channels=1, out_channels=1,
        initial_filters=train_config_model_params.get("initial_filters", 32),
        depth=train_config_model_params.get("depth", 4),
        kernel_size=train_config_model_params.get("kernel_size", 3),
        pool_kernel=train_config_model_params.get("pool_kernel", 2),
        pool_stride=train_config_model_params.get("pool_stride", 2),
        norm_layer=norm_layer_cls,
        activation_fn_class=activation_fn_cls,
        use_residual_conv=train_config_model_params.get("use_residual_conv", True),
        output_activation_class=output_activation_cls,  # This will be None if model outputs logits
        dropout_p=train_config_model_params.get("dropout_p", 0.0)
    )
    try:
        if not model_path or not os.path.exists(model_path):
            utils.logging.error(f"Model loading: File not found at {model_path}")
            return None
        model.load_state_dict(torch.load(model_path, map_location=device))
        utils.logging.info(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        utils.logging.error(f"Error loading model weights from {model_path}: {e}", exc_info=True)
        return None
    model.to(device)
    model.eval()
    return model


# --- Core Data Processing (Candidate for processing.py) ---
# For now, keeping it here as it's tightly coupled with ProcessingThread's current design
def preprocess_for_cnn_and_fitter(
        x_original_unpadded, y_original_unpadded, model_depth, cnn_target_signal_length, cnn_target_x_range
):
    # Ensure inputs are numpy arrays
    x_original_unpadded = np.array(x_original_unpadded, dtype=float)
    y_original_unpadded = np.array(y_original_unpadded, dtype=float)

    if len(y_original_unpadded) == 0 or len(x_original_unpadded) == 0:
        utils.logging.warning("preprocess_for_cnn_and_fitter: Empty input data.")
        # Return a structure that won't cause downstream errors, but indicates emptiness
        dummy_x_cnn_can = np.linspace(cnn_target_x_range[0], cnn_target_x_range[1], cnn_target_signal_length,
                                      dtype=np.float32) if cnn_target_signal_length > 0 else np.array([],
                                                                                                      dtype=np.float32)
        original_cnn_len = len(dummy_x_cnn_can)
        divisor = 2 ** model_depth
        pad_amount_end_cnn = 0
        if original_cnn_len > 0 and original_cnn_len % divisor != 0:
            pad_amount_end_cnn = divisor - (original_cnn_len % divisor)
        dummy_y_padded_norm = np.zeros(original_cnn_len + pad_amount_end_cnn, dtype=np.float32)

        return {"x_cnn_canonical": dummy_x_cnn_can,
                "y_cnn_input_padded_norm": dummy_y_padded_norm,
                "original_cnn_len_unpadded": original_cnn_len,  # Length before padding
                "x_original_unpadded": x_original_unpadded.astype(np.float32),  # Return original as float32
                "y_original_unpadded": y_original_unpadded.astype(np.float32)}

    # Resample to canonical X-axis
    x_cnn_canonical = np.linspace(cnn_target_x_range[0], cnn_target_x_range[1], cnn_target_signal_length,
                                  dtype=np.float32)
    y_resampled_to_cnn_domain = np.zeros_like(x_cnn_canonical, dtype=np.float32)

    if len(x_original_unpadded) == 1:
        y_resampled_to_cnn_domain.fill(y_original_unpadded[0])
    elif len(x_original_unpadded) > 1:
        # Ensure original X is sorted for interpolation
        sort_indices = np.argsort(x_original_unpadded)
        x_orig_sorted, y_orig_sorted = x_original_unpadded[sort_indices], y_original_unpadded[sort_indices]

        # Remove duplicates in x_orig_sorted for interp1d
        unique_x_orig, unique_indices_orig = np.unique(x_orig_sorted, return_index=True)
        if len(unique_x_orig) < 2:  # Need at least two unique points for interp1d
            y_resampled_to_cnn_domain.fill(
                y_orig_sorted[unique_indices_orig[0]] if unique_indices_orig.size > 0 else 0.0)
        else:
            try:
                interp_func = interp1d(unique_x_orig, y_orig_sorted[unique_indices_orig], kind='linear',
                                       fill_value="extrapolate", bounds_error=False)
                y_resampled_to_cnn_domain = interp_func(x_cnn_canonical).astype(np.float32)
            except Exception as e_interp:  # Broader exception for any interp issue
                utils.logging.warning(f"interp1d failed during preprocessing: {e_interp}. Falling back to np.interp.")
                y_resampled_to_cnn_domain = np.interp(x_cnn_canonical, x_orig_sorted, y_orig_sorted).astype(np.float32)

    # Min-Max Normalize Y (0-1 range)
    y_min_for_cnn_norm = np.min(y_resampled_to_cnn_domain)
    y_ptp_for_cnn_norm = np.max(y_resampled_to_cnn_domain) - y_min_for_cnn_norm
    y_norm_for_cnn_unpadded = np.zeros_like(y_resampled_to_cnn_domain, dtype=np.float32)
    if y_ptp_for_cnn_norm >= 1e-7:  # Avoid division by zero or very small range
        y_norm_for_cnn_unpadded = (y_resampled_to_cnn_domain - y_min_for_cnn_norm) / y_ptp_for_cnn_norm

    # Pad to be divisible by 2^depth
    len_before_padding = len(y_norm_for_cnn_unpadded)
    divisor = 2 ** model_depth
    y_cnn_input_padded_norm = y_norm_for_cnn_unpadded
    if len_before_padding > 0 and len_before_padding % divisor != 0:
        pad_amount_end_cnn = divisor - (len_before_padding % divisor)
        y_cnn_input_padded_norm = np.pad(y_norm_for_cnn_unpadded, (0, pad_amount_end_cnn), mode='reflect')

    return {"x_cnn_canonical": x_cnn_canonical,
            "y_cnn_input_padded_norm": y_cnn_input_padded_norm,
            "original_cnn_len_unpadded": len_before_padding,  # Length of y_norm_for_cnn_unpadded
            "x_original_unpadded": x_original_unpadded.astype(np.float32),
            "y_original_unpadded": y_original_unpadded.astype(np.float32)}


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class ProcessingThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, mode, data_args, model_objects, processing_params):
        super().__init__()
        self.mode = mode
        self.data_args = data_args
        self.model_objects = model_objects
        self.processing_params = processing_params
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.progress.emit("Processing cancellation requested...")

    # run_single_sample_processing was provided in the previous response, ensure that corrected version is used
    def run_single_sample_processing(self, x_original_unpadded_arg, y_original_unpadded_arg, sample_name_arg):
        # ... (Use the full, corrected version from the previous response here) ...
        # This is the version that handles all steps and populates `results` correctly.
        # For brevity, not re-pasting the entire ~200 lines, but it's the one that ends with:
        # if not self._is_running: return None
        # return results
        if not self._is_running: return None

        x_original_unpadded = np.array(x_original_unpadded_arg, dtype=float)
        y_original_unpadded = np.array(y_original_unpadded_arg, dtype=float)
        sample_name = sample_name_arg

        model, device, train_config = self.model_objects["cnn_model"], self.model_objects["device"], self.model_objects[
            "train_config"]
        cnn_threshold, cnn_min_distance_prob_map, do_emg_fit, fitter_config_from_gui = \
            self.processing_params["cnn_threshold"], self.processing_params["cnn_min_distance"], \
                self.processing_params["do_emg_fit"], self.processing_params["fitter_config"]

        self.progress.emit(f"Processing {sample_name}: Preprocessing for CNN...")
        model_depth = train_config.get("model_params", {}).get("depth", 4)
        cnn_signal_len = train_config.get("signal_length", 1024)

        # Dynamic cnn_x_range based on input if not well-defined in config, or if config range is too small
        config_x_range = train_config.get("generator_config", {}).get("x_range", None)
        if config_x_range and config_x_range[1] > config_x_range[0]:
            cnn_x_range = config_x_range
        elif len(x_original_unpadded) > 1:
            cnn_x_range = [np.min(x_original_unpadded), np.max(x_original_unpadded)]
            if cnn_x_range[1] <= cnn_x_range[0]: cnn_x_range = [0, 1]  # Fallback for single point data after min/max
        else:  # Single point or empty original data
            cnn_x_range = [0, 1]

        prep_data = preprocess_for_cnn_and_fitter(
            x_original_unpadded, y_original_unpadded, model_depth, cnn_signal_len, cnn_x_range
        )

        if prep_data is None or len(prep_data["y_cnn_input_padded_norm"]) == 0:
            self.error.emit(f"Preprocessing failed for {sample_name}.")
            return None

        if not self._is_running: return None

        input_tensor = torch.from_numpy(prep_data["y_cnn_input_padded_norm"]).float().unsqueeze(0).unsqueeze(0).to(
            device)
        self.progress.emit(f"Processing {sample_name}: CNN Inference...")
        with torch.no_grad():
            pred_tensor = model(input_tensor)
        if not self._is_running: return None

        output_activation_str = train_config.get("model_params", {}).get("output_activation_str", "Sigmoid")
        pred_map_probs_padded = torch.sigmoid(pred_tensor) if output_activation_str == "None" else pred_tensor
        pred_map_probs_cnn_domain_unpadded = pred_map_probs_padded.squeeze().cpu().numpy()[
                                             :prep_data["original_cnn_len_unpadded"]]

        prob_map_on_original_x = np.zeros_like(prep_data["x_original_unpadded"], dtype=float)
        # ... (Interpolation logic as in the last correct version of this function) ...
        if len(prep_data["x_cnn_canonical"]) > 1 and len(prep_data["x_original_unpadded"]) > 0 and len(
                pred_map_probs_cnn_domain_unpadded) > 0:
            sort_indices_cnn_can = np.argsort(prep_data["x_cnn_canonical"])
            x_can_sorted = prep_data["x_cnn_canonical"][sort_indices_cnn_can]

            # Adjust pred_map_sorted based on length of x_can_sorted
            if len(pred_map_probs_cnn_domain_unpadded) == len(x_can_sorted):
                pred_map_sorted = pred_map_probs_cnn_domain_unpadded[sort_indices_cnn_can]
            elif len(pred_map_probs_cnn_domain_unpadded) < len(x_can_sorted):
                pred_map_sorted = np.interp(x_can_sorted,
                                            prep_data["x_cnn_canonical"][sort_indices_cnn_can][
                                            :len(pred_map_probs_cnn_domain_unpadded)],  # x for known y
                                            pred_map_probs_cnn_domain_unpadded[sort_indices_cnn_can[:len(
                                                pred_map_probs_cnn_domain_unpadded)]])  # known y
            else:  # pred_map longer
                pred_map_sorted = pred_map_probs_cnn_domain_unpadded[:len(x_can_sorted)][sort_indices_cnn_can]

            unique_x_can, unique_idx_can = np.unique(x_can_sorted, return_index=True)
            if len(unique_x_can) >= 2:
                interp_func_probs_to_orig = interp1d(unique_x_can, pred_map_sorted[unique_idx_can], kind='linear',
                                                     fill_value="extrapolate", bounds_error=False)
                prob_map_on_original_x = interp_func_probs_to_orig(prep_data["x_original_unpadded"])
            elif len(unique_x_can) == 1:
                prob_map_on_original_x.fill(pred_map_sorted[unique_idx_can[0]])
        elif len(prep_data["x_original_unpadded"]) > 0 and len(pred_map_probs_cnn_domain_unpadded) > 0:
            prob_map_on_original_x.fill(np.mean(pred_map_probs_cnn_domain_unpadded))
        prob_map_on_original_x = np.clip(prob_map_on_original_x, 0.0, 1.0)

        self.progress.emit(f"Processing {sample_name}: Finding peak candidates...")
        detected_indices_on_original_x_cnn, props_cnn = find_peaks(
            prob_map_on_original_x, height=cnn_threshold, distance=cnn_min_distance_prob_map
        )

        cnn_peak_probabilities_at_detection = np.array([])
        if len(detected_indices_on_original_x_cnn) > 0:
            if props_cnn and "peak_heights" in props_cnn:
                sorted_indices_by_prob_height = np.argsort(props_cnn["peak_heights"])[::-1]
                detected_indices_on_original_x_cnn = detected_indices_on_original_x_cnn[sorted_indices_by_prob_height]
                cnn_peak_probabilities_at_detection = props_cnn["peak_heights"][sorted_indices_by_prob_height]
            else:
                cnn_peak_probabilities_at_detection = prob_map_on_original_x[detected_indices_on_original_x_cnn]

        if not self._is_running: return None
        self.progress.emit(
            f"Processing {sample_name}: CNN identified {len(detected_indices_on_original_x_cnn)} candidates.")

        results = {
            "sample_name": sample_name, "x_original": prep_data["x_original_unpadded"],
            "y_original": prep_data["y_original_unpadded"], "x_cnn_canonical": prep_data["x_cnn_canonical"],
            "y_cnn_norm_unpadded": (prep_data["y_cnn_input_padded_norm"][:prep_data["original_cnn_len_unpadded"]]
                                    if prep_data["original_cnn_len_unpadded"] > 0 else np.array([])),
            "pred_map_probs_cnn_domain_unpadded": pred_map_probs_cnn_domain_unpadded,
            "pred_map_probs_on_original_x": prob_map_on_original_x,
            "cnn_raw_detected_indices_original_x": detected_indices_on_original_x_cnn,
            "cnn_raw_detected_probabilities": cnn_peak_probabilities_at_detection,
            "emg_fitter_instance": None, "fitted_peaks_summary": []
        }

        if do_emg_fit and len(detected_indices_on_original_x_cnn) > 0:
            self.progress.emit(
                f"Processing {sample_name}: EMG Fitter init with {len(detected_indices_on_original_x_cnn)} candidates...")
            fitter = EMGFitter(prep_data["x_original_unpadded"], prep_data["y_original_unpadded"],
                               detected_indices_on_original_x_cnn, config=fitter_config_from_gui)
            results["emg_fitter_instance"] = fitter

            if fitter.num_peaks == 0:
                self.progress.emit(f"Processing {sample_name}: No peaks after EMG pre-filters. Skipping curve_fit.")
                for i, cnn_idx in enumerate(detected_indices_on_original_x_cnn):
                    results["fitted_peaks_summary"].append({
                        "peak_id_in_sample": i + 1,
                        "cnn_detected_x_original": prep_data["x_original_unpadded"][cnn_idx],
                        "cnn_detected_prob": results["cnn_raw_detected_probabilities"][i] if i < len(
                            results["cnn_raw_detected_probabilities"]) else prob_map_on_original_x[cnn_idx],
                        "fit_status": "Skipped (Fitter Pre-filtered)",
                        **{k: np.nan for k in
                           ["fit_mu", "fit_sigma", "fit_tau", "fit_amp_scaler", "fit_auc", "fit_auc_percent",
                            "fit_height", "fit_max_pos_x", "fit_fwhm"]}
                    })
            else:
                self.progress.emit(f"Processing {sample_name}: EMG Fitter will attempt {fitter.num_peaks} peaks...")
                fitter.fit()  # fitter.fit_successful is set here

                final_components_params_list = fitter.get_fitted_parameters_list()
                num_final_components = len(final_components_params_list)

                self.progress.emit(
                    f"Processing {sample_name}: EMG curve_fit status: {fitter.fit_successful}. Fitter reported {num_final_components} components.")

                if num_final_components > 0:
                    for i in range(num_final_components):
                        metrics = fitter.calculate_peak_metrics(i)
                        summary_entry = {"peak_id_in_sample": i + 1,
                                         "fit_status": "Successfully Fitted" if fitter.fit_successful and metrics else (
                                             "Fit Attempted (Metrics Failed)" if fitter.fit_successful else "Fit Failed")}
                        if metrics:
                            summary_entry.update(metrics)
                        else:
                            summary_entry.update({k: np.nan for k in
                                                  ["fit_mu", "fit_sigma", "fit_tau", "fit_amp_scaler", "fit_auc",
                                                   "fit_auc_percent", "fit_height", "fit_max_pos_x", "fit_fwhm"]})

                        # Link back to original CNN detection (heuristic)
                        closest_cnn_idx, cnn_x, cnn_prob = -1, np.nan, np.nan
                        if 'fit_mu' in summary_entry and not np.isnan(
                                summary_entry['fit_mu']) and detected_indices_on_original_x_cnn.size > 0:
                            distances = np.abs(
                                prep_data["x_original_unpadded"][detected_indices_on_original_x_cnn] - summary_entry[
                                    'fit_mu'])
                            closest_orig_cnn_idx_in_list = np.argmin(distances)
                            # Heuristic: if distance is < N * sigma of the fitted peak (or some absolute value)
                            if distances[closest_orig_cnn_idx_in_list] < ((summary_entry.get('fit_sigma',
                                                                                             0.1) if summary_entry.get(
                                    'fit_sigma', 0.1) > 0 else 0.1) * 3):  # 3 sigma
                                closest_cnn_idx = detected_indices_on_original_x_cnn[closest_orig_cnn_idx_in_list]
                                cnn_x = prep_data["x_original_unpadded"][closest_cnn_idx]
                                cnn_prob = prob_map_on_original_x[closest_cnn_idx]
                        summary_entry["cnn_detected_x_original"] = cnn_x
                        summary_entry["cnn_detected_prob"] = cnn_prob
                        if summary_entry["fit_status"] == "Successfully Fitted" and closest_cnn_idx == -1:
                            summary_entry["fit_status"] = "Fitted (No clear CNN origin)"
                        results["fitted_peaks_summary"].append(summary_entry)
                elif len(detected_indices_on_original_x_cnn) > 0:  # Fitter had inputs, but 0 final components
                    self.progress.emit(
                        f"Processing {sample_name}: EMG Fitter 0 final components, but had {len(detected_indices_on_original_x_cnn)} initial CNN inputs.")
                    for i, cnn_idx in enumerate(detected_indices_on_original_x_cnn):
                        results["fitted_peaks_summary"].append({
                            "peak_id_in_sample": i + 1,
                            "cnn_detected_x_original": prep_data["x_original_unpadded"][cnn_idx],
                            "cnn_detected_prob": results["cnn_raw_detected_probabilities"][i] if i < len(
                                results["cnn_raw_detected_probabilities"]) else prob_map_on_original_x[cnn_idx],
                            "fit_status": "Not Fitted (EMG Rejected)",
                            **{k: np.nan for k in
                               ["fit_mu", "fit_sigma", "fit_tau", "fit_amp_scaler", "fit_auc", "fit_auc_percent",
                                "fit_height", "fit_max_pos_x", "fit_fwhm"]}
                        })

        elif do_emg_fit and len(detected_indices_on_original_x_cnn) == 0:
            self.progress.emit(f"Processing {sample_name}: No CNN candidates for EMG fitter.")
        elif not do_emg_fit and len(detected_indices_on_original_x_cnn) > 0:
            self.progress.emit(f"Processing {sample_name}: EMG Fitting skipped by user.")
            for i, cnn_idx in enumerate(detected_indices_on_original_x_cnn):
                results["fitted_peaks_summary"].append({
                    "peak_id_in_sample": i + 1,
                    "cnn_detected_x_original": prep_data["x_original_unpadded"][cnn_idx],
                    "cnn_detected_prob": results["cnn_raw_detected_probabilities"][i] if i < len(
                        results["cnn_raw_detected_probabilities"]) else prob_map_on_original_x[cnn_idx],
                    "fit_status": "EMG Fit Skipped (User)",
                    **{k: np.nan for k in
                       ["fit_mu", "fit_sigma", "fit_tau", "fit_amp_scaler", "fit_auc", "fit_auc_percent", "fit_height",
                        "fit_max_pos_x", "fit_fwhm"]}
                })

        if not self._is_running: return None
        return results

    def run(self):  # This method remains largely the same, calling run_single_sample_processing
        try:
            if self.mode == "single_sample":
                if not self._is_running: self.finished.emit(None); return
                result = self.run_single_sample_processing(
                    self.data_args["x_original"], self.data_args["y_original"], self.data_args["sample_name"]
                )
                if self._is_running:
                    self.finished.emit(result)
                else:
                    self.finished.emit(None)

            elif self.mode == "all_samples":
                samples_data_dict = self.data_args["samples_data_dict"]
                sample_names_to_process = self.data_args["sample_names_to_process"]
                all_results_summary_list = []
                total_samples = len(sample_names_to_process)

                for i, s_name in enumerate(sample_names_to_process):
                    if not self._is_running: self.finished.emit(None); return
                    self.progress.emit(f"Batch Processing ({i + 1}/{total_samples}): {s_name}")
                    if s_name not in samples_data_dict:
                        self.error.emit(f"Sample {s_name} not in loaded data. Skipping.");
                        continue

                    x_orig, y_orig = samples_data_dict[s_name]
                    if np.isnan(y_orig).any():  # Should be handled by utils.load_fsec_csv, but double check
                        utils.logging.warning(f"NaNs found in y_orig for {s_name} in batch thread. Using nan_to_num.")
                        y_orig = np.nan_to_num(y_orig,
                                               nan=np.nanmedian(y_orig) if not np.all(np.isnan(y_orig)) else 0.0)

                    current_sample_result_obj = self.run_single_sample_processing(x_orig, y_orig, s_name)
                    if not self._is_running: self.finished.emit(None); return

                    if current_sample_result_obj and current_sample_result_obj.get("fitted_peaks_summary"):
                        for peak_summary in current_sample_result_obj["fitted_peaks_summary"]:
                            peak_summary["sample_name_batch"] = s_name
                        all_results_summary_list.extend(current_sample_result_obj["fitted_peaks_summary"])

                if self._is_running:
                    self.finished.emit(all_results_summary_list)
                else:
                    self.finished.emit(None)
        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if hasattr(e, '__traceback__') and e.__traceback__ else 'N/A'
            error_msg = f"Thread error: {type(e).__name__} - {e} (Line: {tb_lineno})"
            utils.logging.error(error_msg, exc_info=True)
            if self._is_running: self.error.emit(error_msg)
            self.finished.emit(None)


class FSECAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSEC Peak Analyzer v0.7")
        self.setGeometry(50, 50, 1650, 980)  # Slightly wider for legend

        self.loaded_fsec_data = None
        self.all_identified_sample_cols = []
        self.timepoint_col_suggestion = "Timepoint"
        self.current_selected_sample_name = None
        self.cnn_model = None
        self.train_config = None
        self.device = utils.get_device()
        self.model_folder_path = ""
        self.loaded_model_weights_file = ""
        self.csv_path = ""
        self.processing_thread = None
        self.current_sensitivity_settings = {}
        # For table color coding (placeholders, to be configured via GUI/file later)
        self.qc_thresholds = {
            "tau_bad_thresh": 1.5, "tau_warn_thresh": 0.8,
            "main_peak_auc_percent_bad_min": 30.0, "main_peak_auc_percent_warn_min": 60.0,
            "main_peak_auc_percent_good_min": 85.0,
        }

        self.initUI()
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.log_message(f"Application started. Using device: {self.device}", status=True)
        self.update_sensitivity_dependent_params()

    def initUI(self):
        # ... (UI setup from your last full main_gui.py - unchanged visually, ensure all widgets are named as used)
        # For brevity, assuming self.deconv_plot_canvas, self.cnn_view_canvas, self.results_table etc.
        # are all correctly initialized as per your last provided full GUI.
        # The key is that the MplCanvas instances are created.
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_app_layout = QHBoxLayout(main_widget)

        left_panel = QWidget();
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(400);
        left_panel.setMaximumWidth(500)  # Adjusted width
        # --- File Group ---
        file_group = QGroupBox("Load Files");
        file_layout = QGridLayout(file_group)
        self.btn_load_csv = QPushButton("Load FSEC CSV");
        self.btn_load_csv.clicked.connect(self.load_csv_file_triggered)
        self.lbl_csv_path = QLabel("CSV: Not loaded");
        self.lbl_csv_path.setWordWrap(True)
        self.btn_load_model_folder = QPushButton("Load Trained Model Folder");
        self.btn_load_model_folder.clicked.connect(self.load_model_folder_triggered)
        self.lbl_model_folder_path = QLabel("Model Folder: Not loaded");
        self.lbl_model_folder_path.setWordWrap(True)
        file_layout.addWidget(self.btn_load_csv, 0, 0);
        file_layout.addWidget(self.lbl_csv_path, 0, 1, 1, 2)
        file_layout.addWidget(self.btn_load_model_folder, 1, 0);
        file_layout.addWidget(self.lbl_model_folder_path, 1, 1, 1, 2)
        left_layout.addWidget(file_group)
        # --- Sample Group ---
        sample_group = QGroupBox("Sample Selection & View");
        sample_layout = QVBoxLayout(sample_group)
        tp_layout = QHBoxLayout();
        tp_layout.addWidget(QLabel("Timepoint Col:"));
        self.txt_timepoint_col = QLineEdit(self.timepoint_col_suggestion);
        self.txt_timepoint_col.editingFinished.connect(self.update_data_after_timepoint_change);
        tp_layout.addWidget(self.txt_timepoint_col);
        sample_layout.addLayout(tp_layout)
        sample_layout.addWidget(QLabel("Select Sample Column:"));
        self.combo_samples = QComboBox();
        self.combo_samples.currentTextChanged.connect(self.on_sample_selected_changed);
        sample_layout.addWidget(self.combo_samples)
        self.btn_view_all_inputs = QPushButton("View All Input Signals (Plot)");
        self.btn_view_all_inputs.clicked.connect(self.plot_all_input_signals_triggered);
        sample_layout.addWidget(self.btn_view_all_inputs)
        left_layout.addWidget(sample_group)
        # --- Parameters Group ---
        param_group = QGroupBox("Processing Parameters");
        param_layout = QGridLayout(param_group)
        param_layout.addWidget(QLabel("Sensitivity Level:"), 0, 0);
        self.combo_sensitivity = QComboBox();
        self.combo_sensitivity.addItems(["Very Low", "Low", "Medium", "High", "Very High"]);
        self.combo_sensitivity.setCurrentText("Medium");
        self.combo_sensitivity.currentTextChanged.connect(self.update_sensitivity_dependent_params);
        param_layout.addWidget(self.combo_sensitivity, 0, 1)
        param_layout.addWidget(QLabel("CNN Threshold (0-1):"), 1, 0);
        self.txt_cnn_thresh = QLineEdit("0.4");
        self.txt_cnn_thresh.setReadOnly(True);
        param_layout.addWidget(self.txt_cnn_thresh, 1, 1)
        param_layout.addWidget(QLabel("CNN Min Peak Dist (orig. X samples):"), 2, 0);
        self.txt_cnn_min_dist = QLineEdit("3");
        param_layout.addWidget(self.txt_cnn_min_dist, 2, 1)
        self.chk_emg_fit = QCheckBox("Enable EMG Fitting");
        self.chk_emg_fit.setChecked(True);
        param_layout.addWidget(self.chk_emg_fit, 3, 0, 1, 2)
        left_layout.addWidget(param_group)
        # --- Actions Group ---
        action_group = QGroupBox("Actions");
        action_layout = QVBoxLayout(action_group)
        self.btn_predict_single = QPushButton("Analyze Selected Sample");
        self.btn_predict_single.clicked.connect(self.run_single_sample_analysis_triggered);
        action_layout.addWidget(self.btn_predict_single)
        self.btn_process_all = QPushButton("Process All Samples (Batch)");
        self.btn_process_all.clicked.connect(self.run_all_samples_analysis_triggered);
        action_layout.addWidget(self.btn_process_all)
        self.btn_cancel_processing = QPushButton("Cancel Current Processing");
        self.btn_cancel_processing.clicked.connect(self.cancel_processing_triggered);
        self.btn_cancel_processing.setEnabled(False);
        action_layout.addWidget(self.btn_cancel_processing)
        self.progress_bar = QProgressBar();
        self.progress_bar.setVisible(False);
        self.progress_bar.setTextVisible(True);
        action_layout.addWidget(self.progress_bar)
        left_layout.addWidget(action_group)
        # --- Log Output Group ---
        log_group = QGroupBox("Log/Status");
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit();
        self.log_output.setReadOnly(True);
        self.log_output.setMinimumHeight(150);
        log_layout.addWidget(self.log_output)
        left_layout.addWidget(log_group);
        left_layout.addStretch()
        # --- Right Panel ---
        right_panel_widget = QWidget();
        right_main_layout = QHBoxLayout(right_panel_widget)
        plot_area_widget = QWidget();
        plot_area_layout = QVBoxLayout(plot_area_widget)
        plot_area_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.deconv_plot_canvas = MplCanvas(self);
        plot_area_layout.addWidget(QLabel("Deconvolution View:"), alignment=Qt.AlignCenter);
        plot_area_layout.addWidget(self.deconv_plot_canvas, stretch=3)
        self.cnn_view_canvas = MplCanvas(self);
        plot_area_layout.addWidget(QLabel("CNN Processing View:"), alignment=Qt.AlignCenter);
        plot_area_layout.addWidget(self.cnn_view_canvas, stretch=2)
        right_main_layout.addWidget(plot_area_widget, stretch=3)  # Give plots more space than table
        table_area_widget = QWidget();
        table_area_layout = QVBoxLayout(table_area_widget)
        table_area_widget.setMinimumWidth(550);
        table_area_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Increased min width for table
        self.results_table = QTableWidget();
        self.results_table.setAlternatingRowColors(True);
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers);
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows);
        self.results_table.setSelectionMode(QTableWidget.SingleSelection);
        self.results_table.horizontalHeader().setStretchLastSection(True);
        self.results_table.verticalHeader().setVisible(False)  # Stretch last section
        table_area_layout.addWidget(QLabel("Fitted Peak Parameters:"), alignment=Qt.AlignCenter);
        table_area_layout.addWidget(self.results_table, stretch=1)
        right_main_layout.addWidget(table_area_widget, stretch=2)  # Table gets reasonable space
        main_app_layout.addWidget(left_panel);
        main_app_layout.addWidget(right_panel_widget, stretch=1)

    # --- log_message, show_error_message ---
    def log_message(self, message, status=False):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S] ")
        self.log_output.append(timestamp + message)
        self.log_output.ensureCursorVisible()
        if status: self.statusBar.showMessage(message, 7000)
        # QApplication.processEvents() # Remove this to avoid potential issues; draw calls should handle UI updates for plots

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)
        self.log_message(f"ERROR Dialog: {message}")

    # --- CSV and Model Loading ---
    def load_csv_file_triggered(self):
        # ... (Method from previous response, calls _process_loaded_csv_data) ...
        start_dir = os.path.dirname(self.csv_path) if self.csv_path and os.path.exists(
            os.path.dirname(self.csv_path)) else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Load FSEC CSV File", start_dir, "CSV Files (*.csv)")
        if path:
            self.csv_path = path
            try:
                temp_df_for_header = pd.read_csv(path, nrows=0, comment='#')  # Add comment to skip comment lines
                if temp_df_for_header.columns.any():
                    first_col_lower = temp_df_for_header.columns[0].lower().replace(" ", "").replace("_", "")
                    common_time_cols = ["time", "timepoint", "elution", "retention", "volume", "min", "minutes",
                                        "seconds", "sec"]
                    if any(kw in first_col_lower for kw in common_time_cols):
                        self.timepoint_col_suggestion = temp_df_for_header.columns[0]
                    else:
                        self.timepoint_col_suggestion = temp_df_for_header.columns[0]
                    self.txt_timepoint_col.setText(self.timepoint_col_suggestion)
                else:  # No columns found
                    self.show_error_message(f"Could not read columns from CSV: {path}. Is it empty or malformed?")
                    return
            except Exception as e:
                utils.logging.warning(f"Could not peek CSV header for timepoint suggestion: {e}")
            self._process_loaded_csv_data()

    def _process_loaded_csv_data(self):
        # ... (Method from previous response, using utils.load_fsec_csv) ...
        if not self.csv_path: return
        current_timepoint_col = self.txt_timepoint_col.text().strip()
        if not current_timepoint_col: self.show_error_message("Timepoint column name cannot be empty."); return

        loaded_data_dict, identified_cols = utils.load_fsec_csv(self.csv_path, timepoint_col_name=current_timepoint_col)
        if loaded_data_dict:
            self.loaded_fsec_data = loaded_data_dict
            self.all_identified_sample_cols = identified_cols
            self.lbl_csv_path.setText(f"CSV: {os.path.basename(self.csv_path)}")
            self.log_message(
                f"Loaded CSV: {self.csv_path} with timepoint '{current_timepoint_col}' ({len(identified_cols)} samples found).",
                status=True)
            self.populate_sample_dropdown_from_loaded_data()
            if self.all_identified_sample_cols: QTimer.singleShot(100,
                                                                  self.plot_all_input_signals_triggered)  # Plot overview automatically
        else:
            self.loaded_fsec_data = None;
            self.all_identified_sample_cols = []
            self.lbl_csv_path.setText("CSV: Load Error!");
            self.combo_samples.clear();
            self.current_selected_sample_name = None
            self.log_message(
                f"Failed to load data from {self.csv_path} with timepoint '{current_timepoint_col}'. Check logs/CSV format.",
                status=True)
            self._clear_sample_specific_ui()

    def update_data_after_timepoint_change(self):
        # ... (Method from previous response) ...
        if self.csv_path:
            self.log_message(
                f"Timepoint column changed to: '{self.txt_timepoint_col.text().strip()}'. Reloading CSV data.")
            self._process_loaded_csv_data()
        else:
            self.log_message("Timepoint column changed, but no CSV is loaded.")

    def populate_sample_dropdown_from_loaded_data(self):
        # ... (Method from previous response) ...
        self.combo_samples.blockSignals(True);
        self.combo_samples.clear()
        if self.loaded_fsec_data and self.all_identified_sample_cols:
            self.combo_samples.addItems(self.all_identified_sample_cols)
            if self.all_identified_sample_cols:
                # self.combo_samples.setCurrentIndex(0) # This would trigger on_sample_selected_changed
                self.log_message(f"Found {len(self.all_identified_sample_cols)} samples. Defaulting to first.")
            else:
                self.log_message("No valid sample columns found after processing."); self._clear_sample_specific_ui()
        else:
            self.log_message("No data or sample columns to populate dropdown."); self._clear_sample_specific_ui()
        self.combo_samples.blockSignals(False)
        if self.combo_samples.count() > 0:  # If items were added, now trigger selection of the first
            self.combo_samples.setCurrentIndex(0)  # This will now call on_sample_selected_changed

    def _clear_sample_specific_ui(self):
        # ... (Method from previous response) ...
        self.current_selected_sample_name = None
        for canvas in [self.deconv_plot_canvas, self.cnn_view_canvas]:
            if hasattr(canvas, 'fig'):
                canvas.fig.clear()
                ax = canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No data or analysis yet.", ha='center', va='center', color='grey')
                canvas.draw()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)

    def on_sample_selected_changed(self, sample_name):
        # ... (Method from previous response, using visualizations.plot_raw_signals_overview) ...
        if self.loaded_fsec_data and sample_name and sample_name in self.loaded_fsec_data:
            self.current_selected_sample_name = sample_name
            x_data, y_data_raw = self.loaded_fsec_data[sample_name]
            y_data_numeric = np.array(y_data_raw, dtype=float)
            self.log_message(f"Displaying selected raw signal: {sample_name}", status=True)

            try:
                fig, _ = visualizations.plot_raw_signals_overview(
                    {sample_name: (x_data, y_data_numeric)},  # Plot only the selected one
                    timepoint_col_name=self.txt_timepoint_col.text().strip(),
                    title=f"Raw Signal: {sample_name}",
                    max_signals_to_plot=1,
                    highlight_sample=sample_name  # Highlight it
                )
                self.deconv_plot_canvas.fig = fig
                self.deconv_plot_canvas.draw()
            except Exception as e_plot:
                self.log_message(f"Error plotting raw signal for {sample_name}: {e_plot}", status=True)
                utils.logging.error(f"Raw signal plot error for {sample_name}", exc_info=True)
                self.deconv_plot_canvas.fig.clear();
                ax_err = self.deconv_plot_canvas.fig.add_subplot(111);
                ax_err.text(0.5, 0.5, "Plot Error", ha='center');
                self.deconv_plot_canvas.draw()

            if hasattr(self.cnn_view_canvas,
                       'fig'): self.cnn_view_canvas.fig.clear(); ax_cnn_clear = self.cnn_view_canvas.fig.add_subplot(
                111); ax_cnn_clear.text(0.5, 0.5, "Select & Analyze", ha='center'); self.cnn_view_canvas.draw()
            self.results_table.setRowCount(0);
            self.results_table.setColumnCount(0)
        elif sample_name:  # Sample name provided but not in data (e.g., during clear)
            self._clear_sample_specific_ui()

    def plot_all_input_signals_triggered(self):
        # ... (Method from previous response, using visualizations.plot_raw_signals_overview) ...
        if not self.loaded_fsec_data or not self.all_identified_sample_cols:
            self.log_message("No CSV data loaded or no sample columns identified.", status=True);
            self._clear_sample_specific_ui();
            return

        try:
            fig, _ = visualizations.plot_raw_signals_overview(
                self.loaded_fsec_data,
                timepoint_col_name=self.txt_timepoint_col.text().strip(),
                title="All Input Signals Overview",
                highlight_sample=self.current_selected_sample_name  # Highlight current if one is selected
            )
            self.deconv_plot_canvas.fig = fig
            self.deconv_plot_canvas.draw()
        except Exception as e_plot_all:
            self.log_message(f"Error plotting all signals: {e_plot_all}", status=True)
            utils.logging.error("Plot all signals error", exc_info=True)
            self.deconv_plot_canvas.fig.clear();
            ax_err = self.deconv_plot_canvas.fig.add_subplot(111);
            ax_err.text(0.5, 0.5, "Plot Error", ha='center');
            self.deconv_plot_canvas.draw()

        if hasattr(self.cnn_view_canvas,
                   'fig'): self.cnn_view_canvas.fig.clear(); ax_cnn_clear = self.cnn_view_canvas.fig.add_subplot(
            111); ax_cnn_clear.text(0.5, 0.5, "Overview Mode", ha='center'); self.cnn_view_canvas.draw()
        self.results_table.setRowCount(0);
        self.results_table.setColumnCount(0)
        self.log_message(f"Plotted overview of {len(self.loaded_fsec_data)} input signals.", status=True)

    def load_model_folder_triggered(self):
        # ... (Method from previous response, using utils for paths and loading) ...
        start_dir = os.path.dirname(self.model_folder_path) if self.model_folder_path and os.path.exists(
            os.path.dirname(self.model_folder_path)) else os.getcwd()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Trained Model Folder", start_dir)
        if folder_path:
            self.model_folder_path = folder_path
            self.lbl_model_folder_path.setText(f"Folder: ...{os.sep}{os.path.basename(folder_path)}")
            self.log_message(f"Selected model folder: {folder_path}", status=True)
            model_weights_path, config_json_path = utils.get_model_and_config_paths(folder_path)

            if not config_json_path:
                self.show_error_message(f"'config.json' not found in '{folder_path}'.");
                self.train_config = None;
                self.cnn_model = None;
                self.lbl_model_folder_path.setText("Folder: Error - config.json missing!");
                return
            try:
                self.train_config = utils.load_config(config_json_path)
                self.log_message(f"Loaded training config from: {config_json_path}")
            except Exception as e:
                self.show_error_message(f"Error loading config.json: {e}");
                self.train_config = None;
                self.cnn_model = None;
                self.lbl_model_folder_path.setText("Folder: Error - config.json invalid!");
                return
            if not model_weights_path:
                self.show_error_message(f"No suitable '.pth' model found in '{folder_path}'.");
                self.cnn_model = None;
                self.lbl_model_folder_path.setText("Folder: Error - No .pth file!");
                return

            self.log_message(f"Attempting to load weights: {model_weights_path}")
            self.loaded_model_weights_file = os.path.basename(model_weights_path)
            if "model_params" not in self.train_config:
                self.show_error_message("Config missing 'model_params'.");
                self.cnn_model = None;
                self.train_config = None;
                return

            self.cnn_model = load_model_for_gui(model_weights_path, self.train_config["model_params"], self.device)
            if self.cnn_model:
                self.log_message(f"CNN Model '{self.loaded_model_weights_file}' loaded.", status=True);
                self.lbl_model_folder_path.setText(
                    f"Folder: {os.path.basename(folder_path)} ({self.loaded_model_weights_file})")
            else:
                self.show_error_message(
                    f"Failed to load '{self.loaded_model_weights_file}'. Check logs."); self.lbl_model_folder_path.setText(
                    "Folder: Error - .pth load failed!")
        else:
            self.log_message("Model folder selection cancelled.")

    # --- Sensitivity and Parameter Handling ---
    def map_sensitivity_to_params(self, sensitivity_text):
        # ... (Method from previous response) ...
        mapping_threshold = {"Very Low": 0.75, "Low": 0.6, "Medium": 0.4, "High": 0.25, "Very High": 0.1}
        cnn_thresh = mapping_threshold.get(sensitivity_text, 0.4)
        self.current_sensitivity_settings["cnn_threshold"] = cnn_thresh
        self.txt_cnn_thresh.setText(f"{cnn_thresh:.2f}")
        self.log_message(f"Sensitivity set: '{sensitivity_text}', CNN Thresh: {cnn_thresh:.2f}", status=True)

    def update_sensitivity_dependent_params(self):
        # ... (Method from previous response) ...
        self.map_sensitivity_to_params(self.combo_sensitivity.currentText())

    # --- Analysis Triggers and Handlers ---
    def run_single_sample_analysis_triggered(self):
        # ... (Method from previous response, ensures data, model, config are loaded) ...
        if not self.cnn_model or not self.train_config: self.show_error_message("Load Model & Config."); return
        if not self.loaded_fsec_data or not self.current_selected_sample_name: self.show_error_message(
            "Load CSV & select sample."); return
        if self.processing_thread and self.processing_thread.isRunning(): self.show_error_message(
            "Process already running."); return

        x_original, y_original = self.loaded_fsec_data[self.current_selected_sample_name]
        data_args = {"x_original": x_original.copy(), "y_original": y_original.copy(),
                     "sample_name": self.current_selected_sample_name}
        model_objects = {"cnn_model": self.cnn_model, "device": self.device, "train_config": self.train_config}
        try:
            current_cnn_threshold = self.current_sensitivity_settings.get("cnn_threshold",
                                                                          float(self.txt_cnn_thresh.text()))
            cnn_min_dist_val = int(self.txt_cnn_min_dist.text())
            if cnn_min_dist_val < 1: raise ValueError("CNN Min Distance must be >= 1")
            processing_params = {"cnn_threshold": current_cnn_threshold, "cnn_min_distance": cnn_min_dist_val,
                                 "do_emg_fit": self.chk_emg_fit.isChecked(),
                                 "fitter_config": self.train_config.get("emg_fitter_config",
                                                                        {})}  # Use "emg_fitter_config" from train_config
        except ValueError as ve:
            self.show_error_message(f"Invalid parameter: {ve}"); return

        self.log_message(
            f"Starting analysis for: {self.current_selected_sample_name} (Thresh: {processing_params['cnn_threshold']:.2f})",
            status=True)
        self.set_ui_processing_state(True)
        self.processing_thread = ProcessingThread("single_sample", data_args, model_objects, processing_params)
        self.processing_thread.finished.connect(self.on_single_processing_finished)
        self.processing_thread.progress.connect(self.log_message)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()

    def on_single_processing_finished(self, result_obj):
        # --- THIS IS THE METHOD WITH THE CORRECTED AttributeError FIX from PREVIOUS RESPONSE ---
        # Ensure this entire method is the fully corrected one.
        self.set_ui_processing_state(False)
        if result_obj is None:
            self.log_message(
                f"Processing finished with no result for {self.current_selected_sample_name or 'current sample'}.",
                status=True)
            for canvas in [self.deconv_plot_canvas, self.cnn_view_canvas]:
                if hasattr(canvas, 'fig'): canvas.fig.clear(); ax = canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "Processing failed/cancelled.", ha='center', va='center', color='red');
                canvas.draw()
            self.update_results_table([])  # Clear table
            return

        self.log_message(f"Finished analysis for: {result_obj['sample_name']}", status=True)

        # Prepare data for deconvolution plot
        individual_emg_components_data_for_plot = []
        fitter = result_obj.get("emg_fitter_instance")
        num_final_fitter_components = 0
        final_params_list_from_fitter = []
        if fitter:
            final_params_list_from_fitter = fitter.get_fitted_parameters_list()
            num_final_fitter_components = len(final_params_list_from_fitter)

        if self.chk_emg_fit.isChecked() and fitter and fitter.fit_successful and num_final_fitter_components > 0:
            baseline_y_for_components = fitter.get_fitted_baseline_shape()
            component_shapes_above_baseline = fitter.get_individual_peak_shapes()
            for i in range(num_final_fitter_components):
                if i < len(component_shapes_above_baseline) and i < len(final_params_list_from_fitter):
                    component_y_on_baseline = baseline_y_for_components + component_shapes_above_baseline[i]
                    peak_fitter_params = final_params_list_from_fitter[i]
                    label_metrics = {k: peak_fitter_params.get(k) for k in ['mu', 'sigma', 'tau']}
                    # Find corresponding summary to get AUC % for label
                    summary_item_for_label = next((item for item in result_obj.get("fitted_peaks_summary", []) if
                                                   item.get("peak_id_in_sample") == (
                                                               i + 1) and 'fit_mu' in item and np.isclose(
                                                       item['fit_mu'], peak_fitter_params.get("mu", np.nan))), None)
                    if summary_item_for_label and 'fit_auc_percent' in summary_item_for_label:
                        label_metrics['Relative_AUC_Percent'] = summary_item_for_label.get('fit_auc_percent')

                    label = f"Fit {i + 1} ({utils.format_peak_label(label_metrics, include_auc=('Relative_AUC_Percent' in label_metrics), precision=2)})"
                    individual_emg_components_data_for_plot.append({
                        'x_coords': result_obj["x_original"], 'y_coords_on_baseline': component_y_on_baseline,
                        'label': label, 'params': peak_fitter_params})

        # CNN Processing View Plot
        try:
            fig_cnn, _ = visualizations.plot_cnn_processing_view(
                cnn_input_x=result_obj["x_cnn_canonical"],
                cnn_input_y_normalized=result_obj["y_cnn_norm_unpadded"],
                cnn_output_prob_map=result_obj["pred_map_probs_cnn_domain_unpadded"],
                cnn_threshold_value=self.current_sensitivity_settings.get("cnn_threshold"),
                title=f"CNN View: {result_obj['sample_name']}")
            self.cnn_view_canvas.fig = fig_cnn;
            self.cnn_view_canvas.draw()
        except Exception as e:
            utils.logging.error("Error in CNN plot", exc_info=True); self.show_error_message(f"CNN Plot Error: {e}")

        # Deconvolution View Plot
        plot_total_fit_y = fitter.get_total_fit_curve() if fitter else None
        plot_baseline_y = fitter.get_fitted_baseline_shape() if fitter else None
        cnn_detected_x_coords_for_plot = None
        if result_obj.get("cnn_raw_detected_indices_original_x") is not None and result_obj.get(
                "x_original") is not None:
            cnn_detected_x_coords_for_plot = result_obj["x_original"][result_obj["cnn_raw_detected_indices_original_x"]]

        try:
            fig_deconv, _ = visualizations.plot_deconvolution_summary(
                original_x=result_obj["x_original"], original_y=result_obj["y_original"],
                fitted_total_y=plot_total_fit_y,
                individual_emg_components_data=individual_emg_components_data_for_plot if num_final_fitter_components > 0 else None,
                fitted_baseline_y=plot_baseline_y,
                cnn_prob_map_original_x=result_obj["pred_map_probs_on_original_x"],
                cnn_detected_peak_x_coords=cnn_detected_x_coords_for_plot,
                title=f"Deconvolution: {result_obj['sample_name']}")
            self.deconv_plot_canvas.fig = fig_deconv;
            self.deconv_plot_canvas.draw()
        except Exception as e:
            utils.logging.error("Error in Deconv plot", exc_info=True); self.show_error_message(
                f"Deconv Plot Error: {e}")

        # Populate Results Table
        self.update_results_table(result_obj.get("fitted_peaks_summary", []), is_batch_data=False)
        # ... (logging of summary as before) ...

    def update_results_table(self, peak_data_list, is_batch_data=False):
        # ... (Method from previous response - for populating table, potentially with color coding) ...
        # This method should be the one that does NOT include the color coding for now if we are backtracking
        self.results_table.clearContents();
        self.results_table.setRowCount(0)
        if not peak_data_list: self.log_message("No data for results table."); return

        header_map_base = [
            ("Status", "fit_status", "{}"), ("Fit μ", "fit_mu", "{:.3f}"),
            ("Fit σ", "fit_sigma", "{:.3f}"), ("Fit τ", "fit_tau", "{:.4f}"),
            ("Fit Height", "fit_height", "{:.2f}"), ("Fit AUC", "fit_auc", "{:.2f}"),
            ("AUC %", "fit_auc_percent", "{:.1f}%"), ("Fit FWHM", "fit_fwhm", "{:.3f}"),
            ("CNN X", "cnn_detected_x_original", "{:.3f}"), ("CNN Prob", "cnn_detected_prob", "{:.3f}"),
        ]
        active_headers, active_keys_formats = [], []
        if is_batch_data:
            active_headers.extend(["Sample", "Peak #"])
            active_keys_formats.extend([("sample_name_batch", "{}", None), ("peak_id_in_sample", "{:d}", None)])
        else:  # For single sample view, Peak # is more natural first.
            active_headers.append("Peak #")
            active_keys_formats.append(("peak_id_in_sample", "{:d}", None))

        first_item_keys = peak_data_list[0].keys() if peak_data_list else []
        for display_name, data_key, fmt_str in header_map_base:  # Simplified: removed qc_key
            if data_key in first_item_keys:
                active_headers.append(display_name)
                active_keys_formats.append((data_key, fmt_str, None))  # Keep tuple structure, qc_key is None

        if not active_headers: self.log_message("No columns for results table."); return
        self.results_table.setColumnCount(len(active_headers))
        self.results_table.setHorizontalHeaderLabels(active_headers)
        self.results_table.setRowCount(len(peak_data_list))

        for row_idx, peak_data_dict in enumerate(peak_data_list):
            for col_idx, (key, fmt_str, _) in enumerate(active_keys_formats):  # Unpack qc_key as _
                value = peak_data_dict.get(key)
                item_text = "N/A"
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    try:
                        item_text = fmt_str.format(value) if isinstance(value, (int, float, np.number)) else str(value)
                    except:
                        item_text = str(value)
                table_item = QTableWidgetItem(item_text)
                # NO COLOR CODING for now (backtracking)
                if isinstance(value, (int, float, np.number)) and not (isinstance(value, float) and np.isnan(value)):
                    table_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.results_table.setItem(row_idx, col_idx, table_item)

        self.results_table.resizeColumnsToContents()
        # Smart stretching (simplified)
        if self.results_table.columnCount() > 0:
            stretch_col_name = "Sample" if is_batch_data else "Status"
            try:
                stretch_idx = active_headers.index(stretch_col_name)
            except ValueError:
                stretch_idx = self.results_table.columnCount() - 1
            self.results_table.horizontalHeader().setSectionResizeMode(stretch_idx, QHeaderView.Stretch)

    def run_all_samples_analysis_triggered(self):
        # ... (Method from previous response, calls ProcessingThread with "all_samples") ...
        if not self.cnn_model or not self.train_config: self.show_error_message("Load Model & Config."); return
        if not self.loaded_fsec_data or not self.all_identified_sample_cols: self.show_error_message(
            "Load CSV data."); return
        if self.processing_thread and self.processing_thread.isRunning(): self.show_error_message(
            "Process already running."); return

        data_args = {"samples_data_dict": self.loaded_fsec_data,
                     "sample_names_to_process": self.all_identified_sample_cols}
        model_objects = {"cnn_model": self.cnn_model, "device": self.device, "train_config": self.train_config}
        try:
            current_cnn_threshold = self.current_sensitivity_settings.get("cnn_threshold",
                                                                          float(self.txt_cnn_thresh.text()))
            cnn_min_dist_val = int(self.txt_cnn_min_dist.text())
            if cnn_min_dist_val < 1: raise ValueError("CNN Min Distance must be >= 1")
            processing_params = {"cnn_threshold": current_cnn_threshold, "cnn_min_distance": cnn_min_dist_val,
                                 "do_emg_fit": self.chk_emg_fit.isChecked(),
                                 "fitter_config": self.train_config.get("emg_fitter_config", {})}
        except ValueError as ve:
            self.show_error_message(f"Invalid parameter: {ve}"); return

        num_samples = len(self.all_identified_sample_cols)
        self.log_message(f"Starting batch processing for {num_samples} samples...", status=True)
        self.set_ui_processing_state(True);
        self.progress_bar.setVisible(True);
        self.progress_bar.setRange(0, num_samples);
        self.progress_bar.setValue(0)

        self.processing_thread = ProcessingThread("all_samples", data_args, model_objects, processing_params)
        self.processing_thread.finished.connect(self.on_all_processing_finished)
        self.processing_thread.progress.connect(self.update_batch_progress)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()

    def on_all_processing_finished(self, all_results_summary_list):
        # ... (Method from previous response, calls self.update_results_table(all_results_summary_list, is_batch_data=True) and saves CSV)
        self.set_ui_processing_state(False);
        self.progress_bar.setVisible(False)
        if all_results_summary_list is None: self.log_message("Batch processing cancelled or failed.",
                                                              status=True); return
        self.log_message(f"Batch processing complete. {len(all_results_summary_list)} peaks total.", status=True)

        if all_results_summary_list:
            self.update_results_table(all_results_summary_list, is_batch_data=True)  # Update table with batch results
            # CSV Saving logic from before
            batch_col_order = ["sample_name_batch", "peak_id_in_sample", "fit_status", "fit_mu", "fit_sigma", "fit_tau",
                               "fit_height", "fit_auc", "fit_auc_percent", "fit_fwhm", "cnn_detected_x_original",
                               "cnn_detected_prob"]
            csv_basename = os.path.basename(self.csv_path) if self.csv_path else "fsec_data"
            safe_csv_basename = utils.safe_filename(csv_basename.rsplit('.', 1)[0])
            default_filename = f"batch_results_{safe_csv_basename}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            start_dir = os.path.dirname(self.csv_path) if self.csv_path else os.getcwd()
            save_path_suggestion = utils.get_unique_filepath(start_dir, default_filename)
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Batch Results CSV", save_path_suggestion,
                                                       "CSV Files (*.csv)")
            if save_path:
                try:
                    utils.export_results_to_csv(save_path, all_results_summary_list, column_order=batch_col_order)
                except Exception as e:
                    self.show_error_message(f"Error saving batch results: {e}"); utils.logging.error(
                        "Batch CSV save error", exc_info=True)
        else:
            self.update_results_table([], is_batch_data=True); self.log_message(
                "No peak data from batch to save/display.")

    # --- UI State, Cancel, Error, Close ---
    def update_batch_progress(self, message):
        # ... (Method from previous response) ...
        self.log_message(message)
        import re
        match = re.search(r"\((\d+)/(\d+)\)", message)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            if self.progress_bar.maximum() != total: self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        elif "Batch Processing" in message and "/" not in message:
            self.progress_bar.setValue(0)

    def on_processing_error(self, error_message):
        # ... (Method from previous response) ...
        self.set_ui_processing_state(False);
        self.progress_bar.setVisible(False)
        self.show_error_message(f"Processing Error: {error_message}")

    def set_ui_processing_state(self, processing):
        # ... (Method from previous response) ...
        enabled = not processing
        self.btn_load_csv.setEnabled(enabled);
        self.btn_load_model_folder.setEnabled(enabled)
        self.btn_predict_single.setEnabled(enabled);
        self.btn_process_all.setEnabled(enabled)
        self.btn_cancel_processing.setEnabled(processing)
        self.txt_timepoint_col.setEnabled(enabled);
        self.combo_samples.setEnabled(enabled)
        self.combo_sensitivity.setEnabled(enabled);
        self.txt_cnn_min_dist.setEnabled(enabled);
        self.chk_emg_fit.setEnabled(enabled)
        self.progress_bar.setVisible(
            processing != self.progress_bar.isHidden())  # Toggle visibility based on processing state. Fix for progress bar not disappearing

    def cancel_processing_triggered(self):
        # ... (Method from previous response) ...
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message("Attempting to cancel processing...", status=True);
            self.processing_thread.stop()
        else:
            self.log_message("No processing running to cancel.")

    def closeEvent(self, event):
        # ... (Method from previous response) ...
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message("Stopping active processing thread before exit...");
            self.processing_thread.stop()
            if not self.processing_thread.wait(2000): utils.logging.warning(
                "Processing thread did not terminate gracefully.")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Optional: Configure root logger if utils.py doesn't do it sufficiently
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ex = FSECAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())
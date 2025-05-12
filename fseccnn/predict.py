# predict.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import os
import json
import re  # For creating safe filenames from column names

# Assuming these are in the same directory or your PYTHONPATH is set up
from model import UNet1D
from emg_fitter import EMGFitter, emg_peak_scipy  # Import emg_peak_scipy for plotting true components in vis
from cli import parse_predict_args  # Assuming cli.py is correctly defined and provides parse_predict_args


# from synthetic_data import generate_synthetic_chromatogram_mixed_smooth_noise # Only for testing/dev

# Helper to get model layers from config (copied from train.py for self-containment)
def get_norm_activation_layers_pred(config_model_params):
    from torch import nn  # Local import
    norm_layer_map = {"BatchNorm1d": nn.BatchNorm1d, "InstanceNorm1d": nn.InstanceNorm1d, "None": None}
    activation_fn_map = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU}
    output_activation_map = {"Sigmoid": nn.Sigmoid, "None": None}
    norm_layer = norm_layer_map.get(config_model_params.get("norm_layer_str"))  # Use .get for safety
    activation_fn_cls = activation_fn_map.get(config_model_params.get("activation_fn_str"))
    output_activation_cls = output_activation_map.get(config_model_params.get("output_activation_str"))
    return norm_layer, activation_fn_cls, output_activation_cls


def load_model_for_inference(model_path, train_config_model_params, device):
    norm_layer_cls, activation_fn_cls, output_activation_cls = get_norm_activation_layers_pred(
        train_config_model_params)
    # Ensure all necessary params are present with defaults if needed
    model = UNet1D(
        in_channels=1, out_channels=1,  # Standard for this task
        initial_filters=train_config_model_params.get("initial_filters", 32),
        depth=train_config_model_params.get("depth", 4),
        kernel_size=train_config_model_params.get("kernel_size", 3),
        pool_kernel=train_config_model_params.get("pool_kernel", 2),
        pool_stride=train_config_model_params.get("pool_stride", 2),
        norm_layer=norm_layer_cls,
        activation_fn_class=activation_fn_cls,
        use_residual_conv=train_config_model_params.get("use_residual_conv", True),
        output_activation_class=output_activation_cls,
        dropout_p=train_config_model_params.get("dropout_p", 0.0)
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load model weights from {model_path}: {e}")
        return None
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path} and set to evaluation mode.")
    return model


def preprocess_signal_fully_conv(y_original, x_original, model_depth, desired_y_range=(0, 1.0)):
    """
    Pads for divisibility and normalizes Y for fully convolutional inference.
    X-coordinates are also padded for correspondence.
    Returns:
        x_padded (np.array): Padded x-coordinates.
        y_norm_for_cnn (np.array): Padded and normalized y-data for CNN input.
        y_padded_for_fitter (np.array): Padded y-data (original scale) for EMG fitter.
        original_signal_len (int): Length of the original unpadded signal.
    """
    if len(y_original) == 0:
        print("Warning: Empty y_original data for preprocessing.")
        return np.array([]), np.array([]), np.array([]), 0

    divisor = 2 ** model_depth
    original_len = len(y_original)
    y_padded = y_original.astype(np.float32)  # Ensure float for processing
    x_padded = x_original.astype(np.float32)

    pad_amount_end = 0
    if original_len % divisor != 0:
        pad_amount_end = divisor - (original_len % divisor)
        y_padded = np.pad(y_original, (0, pad_amount_end), mode='reflect')
        if len(x_original) > 1:
            last_x_step = x_original[-1] - x_original[-2]
            x_pad_values = x_original[-1] + np.arange(1, pad_amount_end + 1) * last_x_step
        elif len(x_original) == 1:
            x_pad_values = np.full(pad_amount_end, x_original[0])
        else:  # Should not happen if y_original is not empty
            x_pad_values = np.arange(pad_amount_end) * (1.0 / divisor)  # Arbitrary step
        x_padded = np.concatenate([x_original, x_pad_values]) if len(x_original) > 0 else x_pad_values

    # Normalize y_padded (the one that goes into CNN)
    data_min = np.min(y_padded)
    data_ptp = np.max(y_padded) - data_min

    if data_ptp < 1e-7:  # Use a slightly larger epsilon for float comparisons
        y_norm_for_cnn = np.zeros_like(y_padded)
        # data_ptp can remain small, fitter will see y_padded_for_fitter
    else:
        y_norm_for_cnn = (y_padded - data_min) / data_ptp

    # Optional: Scale to a specific range (e.g., if training data had a typical max > 1)
    if desired_y_range != (0, 1.0) and data_ptp >= 1e-7:  # Apply scaling only if not flat
        y_norm_for_cnn = y_norm_for_cnn * (desired_y_range[1] - desired_y_range[0]) + desired_y_range[0]

    y_padded_for_fitter = y_padded  # This is the padded signal on its original scale

    return x_padded.astype(np.float32), y_norm_for_cnn.astype(np.float32), y_padded_for_fitter.astype(
        np.float32), original_len


def safe_filename(name):
    name = str(name)
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '-', name)
    return name if name else "unnamed_sample"


def run_inference_for_sample(
        x_original_unpadded, y_original_unpadded, sample_name, model, device, train_config, cli_args
):
    print(f"\n--- Processing Sample: {sample_name} ---")
    model_depth = train_config["model_params"].get("depth", 4)  # Default depth if not in config

    # Heuristic for desired_y_range based on training data amplitude_range
    # This helps normalize real data to a similar scale as training.
    gen_amp_range = train_config["generator_config"].get("amplitude_range", (0.5, 1.5))
    # Using a fixed range like (0,1) for normalization input to CNN is often simpler and more robust
    # unless you have strong reasons for a different target range for normalized data.
    # For now, let's assume preprocess_signal_fully_conv normalizes to 0-1 for the CNN.
    # The y_padded_for_fitter will retain original scale.

    x_padded_for_input, y_norm_for_cnn, y_padded_for_fitter, original_signal_len = \
        preprocess_signal_fully_conv(y_original_unpadded, x_original_unpadded, model_depth)

    if original_signal_len == 0:
        print(f"Skipping sample {sample_name} due to empty signal after preprocessing.")
        return []

    input_tensor = torch.from_numpy(y_norm_for_cnn).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_tensor = model(input_tensor)

    if train_config["model_params"]["output_activation_str"] == "None":
        pred_map_probs_tensor = torch.sigmoid(pred_tensor)
    else:
        pred_map_probs_tensor = pred_tensor

    pred_map_probs_np_padded = pred_map_probs_tensor.squeeze().cpu().numpy()
    # Trim padding from probability map to match original signal length
    pred_map_probs_np_unpadded = pred_map_probs_np_padded[:original_signal_len]

    cnn_threshold = cli_args.cnn_threshold if cli_args.cnn_threshold is not None else train_config.get(
        "peak_detection_threshold", 0.5)
    cnn_min_distance = cli_args.cnn_min_distance if cli_args.cnn_min_distance is not None else train_config.get(
        "peak_min_distance_samples", 5)

    detected_peak_indices_in_original, properties = find_peaks(
        pred_map_probs_np_unpadded, height=cnn_threshold, distance=cnn_min_distance
    )

    if len(detected_peak_indices_in_original) > 0 and "peak_heights" in properties:
        prominences = properties["peak_heights"]
        sorted_indices = np.argsort(prominences)[::-1]
        detected_peak_indices_in_original = detected_peak_indices_in_original[sorted_indices]

    print(
        f"CNN detected {len(detected_peak_indices_in_original)} potential peaks (Thresh={cnn_threshold:.2f}, Dist={cnn_min_distance}).")

    fitted_peaks_data_for_sample = []
    emg_fitter_instance = None

    if not cli_args.no_emg_fit and len(detected_peak_indices_in_original) > 0:
        print("Attempting EMG deconvolution...")
        fitter_config_cli = train_config.get("emg_fitter_vis_config", {}).copy()  # Use vis_config as base
        if cli_args.fitter_maxfev is not None: fitter_config_cli["maxfev"] = cli_args.fitter_maxfev

        emg_fitter_instance = EMGFitter(
            x_original_unpadded,  # Use original unpadded X
            y_original_unpadded,  # Use original unpadded Y (fitter handles its own scaling needs via guesses)
            detected_peak_indices_in_original,  # Indices are for original_unpadded data
            config=fitter_config_cli
        )
        fit_success = emg_fitter_instance.fit()

        if fit_success:
            print(f"EMG fitting successful for {sample_name}.")
            fitted_params_all = emg_fitter_instance.get_fitted_parameters()
            for i, params_peak_i in enumerate(fitted_params_all):
                # The order of fitted_params_all corresponds to detected_peak_indices_in_original
                # because that's what was passed to the fitter.
                cnn_peak_original_idx = detected_peak_indices_in_original[i]
                metrics = emg_fitter_instance.calculate_peak_metrics(i)
                if metrics:
                    peak_info = {
                        "sample_name": sample_name, "peak_id_in_sample": i + 1,
                        "cnn_detected_x_original": x_original_unpadded[cnn_peak_original_idx],
                        "cnn_detected_prob": pred_map_probs_np_unpadded[cnn_peak_original_idx],
                        "fit_mu": metrics["fitted_mu"], "fit_sigma": metrics["fitted_sigma"],
                        "fit_tau": metrics["fitted_tau"], "fit_amp_scaler": metrics["fitted_A_scaler"],
                        "fit_auc": metrics["auc"], "fit_height": metrics["peak_max_height"],
                        "fit_max_pos_x": metrics["peak_max_position_x"], "fit_fwhm": metrics["fwhm"]
                    }
                    fitted_peaks_data_for_sample.append(peak_info)
        else:
            print(f"EMG fitting failed for {sample_name}.")
    elif cli_args.no_emg_fit:
        print("EMG fitting skipped by user request.")
        for i, cnn_peak_idx in enumerate(detected_peak_indices_in_original):
            fitted_peaks_data_for_sample.append({
                "sample_name": sample_name, "peak_id_in_sample": i + 1,
                "cnn_detected_x_original": x_original_unpadded[cnn_peak_idx],
                "cnn_detected_prob": pred_map_probs_np_unpadded[cnn_peak_idx],
                "fit_mu": np.nan, "fit_sigma": np.nan, "fit_tau": np.nan, "fit_amp_scaler": np.nan,
                "fit_auc": np.nan, "fit_height": np.nan, "fit_max_pos_x": np.nan, "fit_fwhm": np.nan
            })
    else:
        print("No peaks detected by CNN, skipping EMG fitting.")

    if cli_args.save_plot:
        # Determine how many subplots are needed
        num_subplots = 1  # Always show CNN output
        if not cli_args.no_emg_fit and emg_fitter_instance and emg_fitter_instance.fit_successful and emg_fitter_instance.num_peaks > 0:
            num_subplots += 1  # Add space for EMG fit

        fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 5 * num_subplots), sharex=True)
        if num_subplots == 1: axs = [axs]  # Make iterable if only one subplot

        current_ax_idx = 0
        ax_cnn = axs[current_ax_idx]
        ax_cnn.plot(x_original_unpadded, y_original_unpadded, label="Original Signal", color='gray',
                    alpha=0.7)  # Plot original signal for context
        ax_cnn_twin = ax_cnn.twinx()
        ax_cnn_twin.plot(x_original_unpadded, pred_map_probs_np_unpadded, label="CNN Peak Probability", color='purple',
                         alpha=0.7)
        ax_cnn_twin.hlines(cnn_threshold, x_original_unpadded[0], x_original_unpadded[-1], color='red', linestyle=':',
                           label=f'CNN Thresh ({cnn_threshold:.2f})')
        for p_idx in detected_peak_indices_in_original:
            ax_cnn_twin.axvline(x_original_unpadded[p_idx], color='red', linestyle=':', alpha=0.5)
        ax_cnn.set_title(f"CNN Prediction for: {sample_name}")
        ax_cnn.set_ylabel("Original Intensity", color='gray')
        ax_cnn_twin.set_ylabel("CNN Probability", color='purple')
        lines, labels = ax_cnn.get_legend_handles_labels();
        lines2, labels2 = ax_cnn_twin.get_legend_handles_labels()
        ax_cnn.legend(lines + lines2, labels + labels2, loc='upper right');
        ax_cnn.grid(True, alpha=0.3)
        current_ax_idx += 1

        if not cli_args.no_emg_fit and emg_fitter_instance and emg_fitter_instance.fit_successful and emg_fitter_instance.num_peaks > 0:
            ax_emg = axs[current_ax_idx]
            ax_emg.plot(x_original_unpadded, y_original_unpadded, label="Original Data", color='k', alpha=0.6, lw=1.5)
            total_fit_curve = emg_fitter_instance.get_total_fit_curve()
            ax_emg.plot(x_original_unpadded, total_fit_curve, label=f"Total EMG Fit", color='r', ls='--', lw=1.8)
            baseline_curve = emg_fitter_instance.get_fitted_baseline()
            ax_emg.plot(x_original_unpadded, baseline_curve, label="Fitted Baseline", color='magenta', ls=':',
                        alpha=0.9)
            individual_components = emg_fitter_instance.get_individual_peak_shapes()
            colors_emg = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, emg_fitter_instance.num_peaks)))
            for k_fit in range(emg_fitter_instance.num_peaks):
                if k_fit < len(individual_components):
                    ax_emg.fill_between(x_original_unpadded, baseline_curve,
                                        baseline_curve + individual_components[k_fit],
                                        color=colors_emg[k_fit], alpha=0.5, label=f"Fitted Comp. {k_fit + 1}")
            ax_emg.set_title(f"EMG Deconvolution for: {sample_name}")
            ax_emg.set_ylabel("Intensity");
            ax_emg.legend(fontsize='small', ncol=2);
            ax_emg.grid(True, alpha=0.3)
            current_ax_idx += 1

        if num_subplots > 0: axs[-1].set_xlabel("Elution Volume / Time (Original)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        safe_sample_name_file = safe_filename(sample_name)
        plot_out_path = os.path.join(cli_args.output_dir, f"{safe_sample_name_file}_prediction_plot.png")
        plt.savefig(plot_out_path, dpi=200);
        print(f"Plot saved for {sample_name} to: {plot_out_path}");
        plt.close(fig)
    return fitted_peaks_data_for_sample


def main_predict():
    args = parse_predict_args()
    try:
        with open(args.config_path, 'r') as f:
            train_config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Training config file not found: {args.config_path}"); return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from config: {args.config_path}"); return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model = load_model_for_inference(args.model_path, train_config["model_params"], device)
    if model is None: return

    try:
        df = pd.read_csv(args.input_csv, delimiter=args.delimiter, skiprows=args.skip_rows)
    except FileNotFoundError:
        print(f"ERROR: Input CSV not found: {args.input_csv}"); return
    except Exception as e:
        print(f"ERROR loading CSV {args.input_csv}: {e}"); return

    if args.timepoint_col_name not in df.columns:
        print(f"ERROR: Timepoint column '{args.timepoint_col_name}' not found. Available: {df.columns.tolist()}");
        return
    x_common_original = df[args.timepoint_col_name].values.astype(float)

    sample_cols_to_process = []
    if args.sample_column_names:
        for col_name in args.sample_column_names:
            if col_name not in df.columns:
                print(f"Warning: Sample column '{col_name}' not found. Skipping.")
            else:
                sample_cols_to_process.append(col_name)
        if not sample_cols_to_process: print("ERROR: None of specified sample columns found."); return
    else:
        sample_cols_to_process = [col for col in df.columns if col != args.timepoint_col_name]
        if not sample_cols_to_process: print(
            f"ERROR: No data columns found (excluding '{args.timepoint_col_name}')."); return
        print(f"Processing all found sample columns: {len(sample_cols_to_process)}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_samples_fitted_data = []

    for sample_col_name in sample_cols_to_process:
        y_current_original = df[sample_col_name].values  # astype float done in preprocess
        nan_mask = pd.isna(y_current_original)  # Use pandas isna for broader NaN detection
        if np.any(nan_mask):
            print(
                f"Warning: Found {np.sum(nan_mask)} NaN/missing value(s) in sample '{sample_col_name}'. Attempting to interpolate...")
            y_current_original = y_current_original.astype(float)  # Ensure float before interp
            not_nan_mask = ~nan_mask
            if np.sum(not_nan_mask) < 2:
                print(f"ERROR: Too few non-NaN data points in '{sample_col_name}' to process. Skipping.")
                continue
            # Interpolate NaNs using non-NaN points
            y_current_original[nan_mask] = np.interp(
                x_common_original[nan_mask], x_common_original[not_nan_mask], y_current_original[not_nan_mask]
            )
            if np.any(np.isnan(
                    y_current_original)):  # Still NaNs after interp (e.g. all NaNs, or leading/trailing outside interp range)
                print(
                    f"ERROR: Could not fully interpolate NaNs in '{sample_col_name}'. Filling remaining with 0. Caution advised.")
                y_current_original = np.nan_to_num(y_current_original, nan=0.0)  # Replace any remaining NaNs with 0

        fitted_data = run_inference_for_sample(
            x_common_original.copy(), y_current_original.copy(), sample_col_name,  # Pass copies
            model, device, train_config, args
        )
        if fitted_data: all_samples_fitted_data.extend(fitted_data)

    if args.save_results_csv and all_samples_fitted_data:
        overall_results_df = pd.DataFrame(all_samples_fitted_data)
        base_input_filename = os.path.splitext(os.path.basename(args.input_csv))[0]
        csv_out_path = os.path.join(args.output_dir, f"{base_input_filename}_ALL_SAMPLES_fseccnn_results.csv")
        overall_results_df.to_csv(csv_out_path, index=False)
        print(f"\nAll fitted peak parameters saved to: {csv_out_path}")
    elif args.save_results_csv:
        print("\nNo peaks were successfully processed to save detailed CSV results.")
    print("\nPrediction script finished.")


if __name__ == "__main__":
    main_predict()
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  # For post-processing predictions
import time
import os
import json  # For saving config

# Import your custom modules
from synthetic_data import (
    LargeSyntheticFSECDataset,
    generate_synthetic_chromatogram_mixed_smooth_noise,  # Used by create_large_offline_dataset & visualization
    create_large_offline_dataset
)
from model import UNet1D
from emg_fitter import EMGFitter, emg_peak_scipy  # Assuming emg_peak_scipy is module level for direct use
from losses import FocalLoss, DiceLoss, CombinedLoss, FocalBCELoss  # Import custom losses

# --- Configuration & Hyperparameters ---
GENERATOR_FUNCTION_MAP = {  # Map string names to actual function objects
    "generate_synthetic_chromatogram_mixed_smooth_noise": generate_synthetic_chromatogram_mixed_smooth_noise
}

# --- Configuration & Hyperparameters ---
def get_config():
    config = {
        "experiment_name": "FSEC_UNet1D_FocalBCE_v1",  # Example: Updated experiment name
        "run_timestamp": time.strftime("%Y%m%d-%H%M%S"),

        # Data parameters
        "offline_dataset_path": "synthetic_data/fsec_offline_dataset_main.h5",
        "num_total_samples_offline": 100000,
        "offline_creation_chunk_size": 10000,
        "epoch_subset_size": 20000,
        "val_epoch_subset_size": 2000,
        "signal_length": 256,

        "generator_fn_str": "generate_synthetic_chromatogram_mixed_smooth_noise",
        "generator_config": {
            'length': 256, 'x_range': (0, 100), 'max_peaks': 4, 'min_peaks': 1,
            'amplitude_range': (0.6, 1.4), 'std_dev_range': (2.0, 7.0),
            'tau_range': (0.0, 1.5), 'peak_type_ratio': 0.6,
            'shoulder_peak_probability': 0.5,
            'shoulder_amplitude_ratio_range': (0.15, 0.4),
            'shoulder_std_dev_ratio_range': (0.4, 1.0),
            'shoulder_mu_offset_ratio_range': (-1.5, 1.5),
            'shoulder_tau_factor_range': (0.0, 1.0),
            'base_noise_level': 0.004,
            'noise_smoothing_window_size': 9,
            'noise_smoothing_polyorder': 2,
            'baseline_linear_drift_variation': 0.1,
            'baseline_wobble_amplitude_factor': 0.04,
            'baseline_wobble_periods_range': (0.4, 1.5),
            'baseline_y_corrected_min_level': -0.02
        },

        # Model parameters
        "model_params": {
            "initial_filters": 32, "depth": 4, "kernel_size": 3,
            "pool_kernel": 2, "pool_stride": 2, "norm_layer_str": "BatchNorm1d",
            "activation_fn_str": "ReLU", "use_residual_conv": True,
            "output_activation_str": "None",  # Important: "None" for logits, which FocalLoss & BCEWithLogitsLoss expect
            "dropout_p": 0.1
        },

        # Training parameters
        "num_epochs": 500, "batch_size": 64, "learning_rate": 3e-4,
        "optimizer": "AdamW", "weight_decay": 1e-5, "scheduler": "ReduceLROnPlateau",
        "scheduler_params": {
            "ReduceLROnPlateau": {"mode": 'min', "factor": 0.2, "patience": 20},
            "StepLR": {"step_size": 30, "gamma": 0.1},
            "CosineAnnealingLR": {"T_max_epochs": 100, "eta_min_factor": 0.01}
        },

        "loss_function": "FocalBCELoss", # Set to your desired default loss
        "loss_params": {
            # --- Defaults for individual loss components ---
            "FocalLoss": {"alpha": 0.25, "gamma": 2.0, "reduction": "mean"},
            "DiceLoss": {"smooth": 1e-6, "reduction": "mean"}, # reduction not strictly used by Dice but for consistency
            "BCEWithLogitsLoss": {"reduction": "mean"}, # Can add e.g. "pos_weight": torch.tensor([SOME_VALUE])
            "BCELoss": {"reduction": "mean"},

            # --- Configuration for CombinedLoss (e.g., Focal/BCE + Dice) ---
            "CombinedLoss": {
                "loss1_type": "FocalLoss",   # Options: "FocalLoss", "BCEWithLogitsLoss", "BCELoss"
                "loss1_weight": 0.7,
                # loss1_params will be merged from above defaults and specific overrides here
                "loss1_params": {"alpha": 0.25, "gamma": 2.0}, # Example override for FocalLoss
                "dice_weight": 0.3,
                "dice_params": {"smooth": 1e-5}                # Example override for DiceLoss
            },

            # --- Configuration for FocalBCELoss (Focal + BCE variant) ---
            "FocalBCELoss": {
                "focal_weight": 0.3,
                # focal_params will be merged from "FocalLoss" defaults and specific overrides here
                "focal_params": {"alpha": 0.3, "gamma": 1.8}, # Example: Specific params for Focal in this combo

                "bce_weight": 0.7,
                "bce_loss_type": "BCEWithLogitsLoss", # Choose 'BCEWithLogitsLoss' or 'BCELoss'
                                                      # Ensure this is compatible with model_params["output_activation_str"]
                                                      # and the inputs_are_logits flag in FocalBCELoss constructor.
                # bce_params will be merged from "BCEWithLogitsLoss" or "BCELoss" defaults and specific overrides
                "bce_params": {} # Example: could be {"pos_weight": torch.tensor([2.0])} for BCEWithLogitsLoss if needed
                                 # Make sure to import torch if using torch.tensor here.
            }
        },
        "early_stopping_patience": 100,

        # Post-processing & EMG Fitter
        "peak_detection_threshold": 0.4, "peak_min_distance_samples": 3,
        "emg_fitter_vis_config": {
            "maxfev": 20000, "sigma_guess_fraction": 0.04, "tau_guess_factor": 0.05
        },

        # System parameters
        "num_workers_dataloader": 0, "log_dir_base": "runs_fsec_offline",
        "save_model_dir_base": "trained_fsec_models_offline",
        "log_images_every_n_epochs": 10, "seed": 42,
    }
    # Ensure generator_config 'length' matches main 'signal_length'
    config["generator_config"]["length"] = config["signal_length"]
    return config

CONFIG = get_config() # This line would typically be outside the function if CONFIG is global


# --- Utility Functions ---
def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_norm_activation_layers(config_model):
    norm_layer_map = {"BatchNorm1d": nn.BatchNorm1d, "InstanceNorm1d": nn.InstanceNorm1d, "None": None}
    activation_fn_map = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU}
    output_activation_map = {"Sigmoid": nn.Sigmoid, "None": None}
    norm_layer = norm_layer_map.get(config_model["norm_layer_str"])
    activation_fn = activation_fn_map.get(config_model["activation_fn_str"])
    output_activation = output_activation_map.get(config_model["output_activation_str"])
    return norm_layer, activation_fn, output_activation


# --- Training & Evaluation Functions ---
def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num, writer, num_total_batches):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad();
        outputs = model(inputs);
        loss = criterion(outputs, targets)
        loss.backward();
        optimizer.step();
        running_loss += loss.item()
        if writer and (i % max(1, num_total_batches // 20) == 0 or i == num_total_batches - 1):
            writer.add_scalar('Loss/train_batch', loss.item(), epoch_num * num_total_batches + i)
    return running_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval();
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs);
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataloader)

# --- Visualization Function (with EMG Fit) ---
def visualize_predictions_with_emg_fit(model, device, config, epoch_num, writer, num_samples_to_show=1):
    model.eval()
    # Ensure matplotlib backend is suitable for non-interactive environments if running on a server
    # import matplotlib
    # matplotlib.use('Agg') # Uncomment if running in a headless environment and getting display errors

    fig_cnn, axs_cnn = plt.subplots(num_samples_to_show, 3, figsize=(18, 4 * num_samples_to_show), sharex='col')
    if num_samples_to_show == 1: axs_cnn = np.expand_dims(axs_cnn,
                                                          axis=0)  # Ensure axs_cnn is always 2D for consistent indexing
    fig_cnn.suptitle(f"CNN Predictions - Epoch: {epoch_num}", fontsize=16)

    first_sample_data_for_emg = None  # To store data for the first sample's EMG fit plot
    generator_fn = GENERATOR_FUNCTION_MAP.get(config["generator_fn_str"])

    if generator_fn is None:
        print(f"ERROR: Generator function '{config['generator_fn_str']}' not found in GENERATOR_FUNCTION_MAP.")
        plt.close(fig_cnn)
        return

    model_outputs_logits_for_vis = config["model_params"]["output_activation_str"] == "None"

    for i in range(num_samples_to_show):
        # Generate a synthetic sample
        # The signature of your generator_fn might vary. Adjust if it doesn't return these exact 4 values.
        y_signal_np, target_map_np, true_peak_params_list, x_coords_np, *_ = generator_fn(**config["generator_config"])
        # Ensure input to model is (Batch, Channels, Length)
        signal_tensor = torch.from_numpy(y_signal_np).float().unsqueeze(0).unsqueeze(0).to(device)

        if i == 0:  # Store data from the first sample for detailed EMG fitting visualization
            first_sample_data_for_emg = {
                "x_coords": x_coords_np,
                "y_signal": y_signal_np,
                "true_peak_params_list": true_peak_params_list
            }

        # Get model prediction
        with torch.no_grad():
            pred_tensor = model(signal_tensor)

        # Process prediction (apply sigmoid if logits, then to numpy)
        pred_map_np = torch.sigmoid(
            pred_tensor).squeeze().cpu().numpy() if model_outputs_logits_for_vis else pred_tensor.squeeze().cpu().numpy()

        # Find peaks in prediction map
        detected_peak_indices_cnn, _ = find_peaks(pred_map_np, height=config["peak_detection_threshold"],
                                                  distance=config["peak_min_distance_samples"])
        # Find true target maxima for visualization
        true_target_maxima_indices = []
        if np.any(target_map_np > 0.5):  # Check if there are any target regions
            # Find connected regions where target_map_np > 0.5
            regions = np.split(np.where(target_map_np > 0.5)[0],
                               np.where(np.diff(np.where(target_map_np > 0.5)[0]) > 1)[0] + 1)
            for region in regions:
                if len(region) > 0:
                    true_target_maxima_indices.append(int(np.mean(region)))  # Center of the target region

        # Plotting CNN results
        axs_cnn[i, 0].plot(x_coords_np, y_signal_np, label="Input Signal", color='blue', alpha=0.8)
        label_true_target = 'True Target Max' if i == 0 and true_target_maxima_indices else None
        label_detected_cnn = 'CNN Detected Peak' if i == 0 and detected_peak_indices_cnn.size > 0 else None

        for idx_iter, idx_true in enumerate(true_target_maxima_indices):
            axs_cnn[i, 0].axvline(x_coords_np[idx_true], color='g', linestyle='--', alpha=0.6,
                                  label=label_true_target if idx_iter == 0 else None)
        for idx_iter, idx_cnn in enumerate(detected_peak_indices_cnn):
            axs_cnn[i, 0].axvline(x_coords_np[idx_cnn], color='r', linestyle=':', alpha=0.7,
                                  label=label_detected_cnn if idx_iter == 0 else None)
        axs_cnn[i, 0].set_ylabel("Intensity")
        if i == 0: axs_cnn[i, 0].set_title("Input & CNN Detections"); axs_cnn[i, 0].legend(fontsize='small')

        axs_cnn[i, 1].plot(x_coords_np, target_map_np, label="Ground Truth Map", color='green')
        if i == 0: axs_cnn[i, 1].set_title("GT Map (CNN Target)"); axs_cnn[i, 1].legend(fontsize='small')

        axs_cnn[i, 2].plot(x_coords_np, pred_map_np, label="Predicted Prob. Map", color='purple')
        axs_cnn[i, 2].hlines(config["peak_detection_threshold"], x_coords_np[0], x_coords_np[-1], colors='gray',
                             linestyles='--',
                             label=f'Threshold ({config["peak_detection_threshold"]:.2f})' if i == 0 else None)
        if i == 0: axs_cnn[i, 2].set_title("Predicted Map (CNN Output)"); axs_cnn[i, 2].legend(fontsize='small')

        if i == num_samples_to_show - 1:  # Add x-labels only to the last row
            axs_cnn[i, 0].set_xlabel("Elution Index / Time")
            axs_cnn[i, 1].set_xlabel("Elution Index / Time")
            axs_cnn[i, 2].set_xlabel("Elution Index / Time")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle
    if writer:
        writer.add_figure(f'CNN_Predictions/Epoch_{epoch_num}', fig_cnn,
                          global_step=epoch_num if isinstance(epoch_num, int) else config[
                              "num_epochs"])  # Use config for "final"
    else:
        fig_cnn.show()  # For local debugging if writer is None
    plt.close(fig_cnn)  # Close the figure to free memory

    # --- EMG Fit Visualization on the first sample ---
    if first_sample_data_for_emg:
        x_coords_fit = first_sample_data_for_emg["x_coords"]
        y_signal_fit = first_sample_data_for_emg["y_signal"]
        true_peak_params_list_fit = first_sample_data_for_emg["true_peak_params_list"]

        # Re-run CNN prediction for the sample chosen for EMG fit (could reuse if stored, but this is safer)
        signal_tensor_fit = torch.from_numpy(y_signal_fit).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_tensor_fit = model(signal_tensor_fit)
        pred_map_np_fit = torch.sigmoid(
            pred_tensor_fit).squeeze().cpu().numpy() if model_outputs_logits_for_vis else pred_tensor_fit.squeeze().cpu().numpy()

        # Get peak indices from CNN prediction for the fitter
        detected_indices_for_fitter, _ = find_peaks(pred_map_np_fit, height=config["peak_detection_threshold"],
                                                    distance=config["peak_min_distance_samples"])

        # Ensure indices are valid and within bounds of x_coords_fit
        valid_detected_indices = [idx for idx in detected_indices_for_fitter if 0 <= idx < len(x_coords_fit)]

        if not valid_detected_indices and true_peak_params_list_fit:  # If CNN finds no peaks, use true peaks as fallback for vis
            # Fallback to true peak locations if CNN fails for visualization purposes
            # This helps debug EMG fitter even if CNN is performing poorly initially
            valid_detected_indices = []
            for p_true in true_peak_params_list_fit:
                # Find index in x_coords_fit closest to true mu
                if 'mu' in p_true:
                    true_mu_idx = np.argmin(np.abs(x_coords_fit - p_true['mu']))
                    if 0 <= true_mu_idx < len(x_coords_fit):
                        valid_detected_indices.append(true_mu_idx)
            valid_detected_indices = sorted(list(set(valid_detected_indices)))  # Remove duplicates and sort
            print(
                f"Warning: CNN found no peaks for EMG visualization (Epoch {epoch_num}). Falling back to true peak locations for EMG fitter.")
        elif not valid_detected_indices:
            print(
                f"Warning: No peaks (neither CNN nor true fallback) for EMG fitter visualization (Epoch {epoch_num}). Skipping EMG plot.")
            return  # Skip EMG plot if no indices

        # Initialize and run EMGFitter
        emg_fitter_instance = EMGFitter(x_coords_fit, y_signal_fit, valid_detected_indices,
                                        config=config.get("emg_fitter_vis_config", {}))  # Use vis_config or empty dict
        fit_success_vis = emg_fitter_instance.fit()

        # Plot EMG Fitter results
        fig_emg, ax_emg = plt.subplots(1, 1, figsize=(12, 7))
        ax_emg.plot(x_coords_fit, y_signal_fit, label="Original Data", color='black', alpha=0.6, linewidth=1.5)

        # Get fitted components from the fitter AFTER fit() and any post-filtering inside fitter
        final_fitted_components = emg_fitter_instance.get_individual_peak_shapes()
        baseline_curve = emg_fitter_instance.get_fitted_baseline_shape()  # <<< CORRECTED METHOD NAME
        total_fit_curve = emg_fitter_instance.get_total_fit_curve()

        num_actual_fitted_peaks = len(final_fitted_components)

        if num_actual_fitted_peaks > 0 or np.any(baseline_curve):  # If there are peaks or at least a baseline
            ax_emg.plot(x_coords_fit, total_fit_curve, label=f"Total EMG Fit (Success: {fit_success_vis})", color='red',
                        linestyle='--', linewidth=1.5)
            ax_emg.plot(x_coords_fit, baseline_curve, label="Fitted Baseline", color='purple', linestyle=':', alpha=0.8)

            if num_actual_fitted_peaks > 0:
                colors_emg = plt.cm.viridis(np.linspace(0.1, 0.9, num_actual_fitted_peaks))
                for k in range(num_actual_fitted_peaks):
                    # Plot each component on top of the fitted baseline
                    ax_emg.fill_between(x_coords_fit, baseline_curve,
                                        baseline_curve + final_fitted_components[k],
                                        color=colors_emg[k] if num_actual_fitted_peaks > 1 else colors_emg[0],
                                        # Handle single peak color
                                        alpha=0.5,
                                        label=f"Fitted Comp. {k + 1}")
        else:  # No peaks were fitted and baseline might be zero
            ax_emg.text(0.5, 0.5, "No components fitted by EMGFitter.", ha='center', va='center',
                        transform=ax_emg.transAxes)

        # Plot true components for comparison if available
        if true_peak_params_list_fit:
            # Assuming synthetic data might have a true baseline offset encoded
            true_baseline_offset_for_plot = 0.0
            # If your true_peak_params_list_fit items contain baseline info, extract it
            # For example, if the first true peak's dict has 'baseline_offset'
            if true_peak_params_list_fit and 'baseline_offset' in true_peak_params_list_fit[0]:
                true_baseline_offset_for_plot = true_peak_params_list_fit[0]['baseline_offset']

            true_baseline_for_vis = np.full_like(x_coords_fit, true_baseline_offset_for_plot)

            colors_true_vis = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(true_peak_params_list_fit)))
            for k_true, p_true in enumerate(true_peak_params_list_fit):
                tau_true_vis = p_true.get('tau', 1e-7)  # Use small tau for Gaussian if 'tau' not present
                # Use 'A_scaler' if that's what your synthetic generator provides, or 'A' for amplitude
                amp_true_vis = p_true.get('A_scaler', p_true.get('A', 1.0))  # Fallback to A or 1.0

                true_comp_vis = emg_peak_scipy(x_coords_fit, amp_true_vis, p_true['mu'], p_true['sigma'], tau_true_vis)
                # Plot true component on its own true baseline (if known)
                ax_emg.plot(x_coords_fit, true_comp_vis + true_baseline_for_vis, color=colors_true_vis[k_true],
                            linestyle=':',
                            alpha=0.9, linewidth=1.2,
                            label=f"True Comp. {k_true + 1} {'(S)' if p_true.get('is_shoulder') else ''}")

        ax_emg.set_title(f"EMG Fit Attempt on Sample - Epoch: {epoch_num}")
        ax_emg.set_xlabel("Elution Index / Time")
        ax_emg.set_ylabel("Intensity")

        # Dynamic legend columns
        handles, labels = ax_emg.get_legend_handles_labels()
        if handles:
            ax_emg.legend(handles, labels, fontsize='small', ncol=max(1, len(handles) // 4))  # Auto-columns
        ax_emg.grid(True, linestyle=':', alpha=0.5)

        # Adjust y-limits to ensure visibility
        if len(y_signal_fit) > 0:
            min_val = np.min(y_signal_fit)
            max_val = np.max(y_signal_fit)
            if num_actual_fitted_peaks > 0 or np.any(baseline_curve):
                min_val = min(min_val, np.min(total_fit_curve))
                max_val = max(max_val, np.max(total_fit_curve))
            y_range = max_val - min_val if max_val > min_val else 1.0
            ax_emg.set_ylim(min_val - 0.1 * y_range, max_val + 0.1 * y_range)

        fig_emg.tight_layout()
        if writer:
            writer.add_figure(f'EMG_Fit_Details/Epoch_{epoch_num}', fig_emg,
                              global_step=epoch_num if isinstance(epoch_num, int) else config["num_epochs"])
        else:
            fig_emg.show()
        plt.close(fig_emg)


def main():
    set_seed(CONFIG["seed"])  # Use the global CONFIG

    # Setup Directories & Logging
    run_dir = os.path.join(CONFIG["log_dir_base"], f"{CONFIG['experiment_name']}_{CONFIG['run_timestamp']}")
    model_save_dir = os.path.join(CONFIG["save_model_dir_base"],
                                  f"{CONFIG['experiment_name']}_{CONFIG['run_timestamp']}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)
    with open(os.path.join(model_save_dir, "config.json"), 'w') as f:  # Also save with model
        json.dump(CONFIG, f, indent=4)

    writer = SummaryWriter(log_dir=run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Experiment: {CONFIG['experiment_name']} ---")
    print(f"Timestamp: {CONFIG['run_timestamp']}")
    print(f"Using device: {device}")
    print(f"Logging to: {run_dir}")
    print(f"Models will be saved to: {model_save_dir}")

    # --- Prepare Offline Dataset (Generate if not exists) ---
    print("\nChecking for offline HDF5 dataset...")
    if not os.path.exists(CONFIG["offline_dataset_path"]):
        print(f"Offline dataset not found at {CONFIG['offline_dataset_path']}.")
        user_choice = input(
            f"Do you want to generate it now with {CONFIG['num_total_samples_offline']} samples? (y/N): ").strip().lower()
        if user_choice == 'y':
            print(f"Generating offline dataset. This may take a while...")
            generator_fn = GENERATOR_FUNCTION_MAP.get(CONFIG["generator_fn_str"])
            if generator_fn is None:
                print(f"ERROR: Generator function '{CONFIG['generator_fn_str']}' not found in GENERATOR_FUNCTION_MAP.")
                writer.close()
                return
            create_large_offline_dataset(
                filepath=CONFIG["offline_dataset_path"],
                num_total_samples=CONFIG["num_total_samples_offline"],
                generator_fn=generator_fn,
                generator_config=CONFIG["generator_config"],
                chunk_size=CONFIG["offline_creation_chunk_size"]
            )
            print("Offline dataset generation complete.")
        else:
            print("Exiting. Please generate the offline dataset first or provide the correct path.")
            writer.close()
            return
    else:
        print(f"Found offline dataset at {CONFIG['offline_dataset_path']}.")

    # Datasets and DataLoaders
    print("\nInitializing datasets from offline HDF5...")
    train_dataset = LargeSyntheticFSECDataset(CONFIG["offline_dataset_path"], CONFIG["epoch_subset_size"])
    val_dataset = LargeSyntheticFSECDataset(CONFIG["offline_dataset_path"],
                                            CONFIG["val_epoch_subset_size"])  # Or a separate val HDF5
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=False,  # Shuffle handled by dataset.reshuffle_indices
        num_workers=CONFIG["num_workers_dataloader"], pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers_dataloader"], pin_memory=(device.type == 'cuda')
    )
    print(f"Offline datasets ready. Train subset size: {len(train_dataset)}, Val subset size: {len(val_dataset)}")

    # Model
    print("\nInitializing model...")
    norm_layer_cls, activation_fn_cls, output_activation_cls = get_norm_activation_layers(CONFIG["model_params"])
    model_cfg = CONFIG["model_params"]
    model = UNet1D(
        in_channels=1, out_channels=1,
        initial_filters=model_cfg["initial_filters"],
        depth=model_cfg["depth"],
        kernel_size=model_cfg["kernel_size"],
        pool_kernel=model_cfg["pool_kernel"],
        pool_stride=model_cfg["pool_stride"],
        norm_layer=norm_layer_cls,
        activation_fn_class=activation_fn_cls,  # Using corrected keyword
        use_residual_conv=model_cfg["use_residual_conv"],
        output_activation_class=output_activation_cls,  # Using corrected keyword
        dropout_p=model_cfg.get("dropout_p", 0.0)
    ).to(device)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Loss Function Instantiation
    criterion = None
    loss_name = CONFIG["loss_function"]
    # This flag is determined by model's output activation: True if logits, False if probabilities (e.g. Sigmoid)
    model_outputs_logits = CONFIG["model_params"]["output_activation_str"] == "None"

    print(f"\nInitializing loss function: {loss_name} (Model outputs logits: {model_outputs_logits})")

    if loss_name == "BCEWithLogitsLoss":
        if not model_outputs_logits:
            print("Warning: BCEWithLogitsLoss expects logits, but model output_activation is not None.")
        criterion = nn.BCEWithLogitsLoss(**CONFIG["loss_params"].get("BCEWithLogitsLoss", {})).to(device)
    elif loss_name == "BCELoss":
        if model_outputs_logits:
            print(
                "Warning: BCELoss expects probabilities, but model output_activation is None. Will apply sigmoid before loss.")
            # This case needs careful handling or model should output probabilities
            # For simplicity, let's assume if BCELoss is chosen, model_outputs_logits is False,
            # or we add a wrapper. The CombinedLoss/FocalBCELoss handle this.
            # If user wants standalone BCELoss with logits model, they need to be careful.
        criterion = nn.BCELoss(**CONFIG["loss_params"].get("BCELoss", {})).to(device)
        # If model_outputs_logits is True for standalone BCELoss, the user should add a sigmoid layer to the model
        # or use BCEWithLogitsLoss instead.
    elif loss_name == "FocalLoss":
        if not model_outputs_logits:
            print("Warning: FocalLoss expects logits, but model output_activation is not None.")
        criterion = FocalLoss(**CONFIG["loss_params"].get("FocalLoss", {})).to(device)
    elif loss_name == "DiceLoss":
        # DiceLoss typically expects probabilities.
        # If model outputs logits, a sigmoid is needed. This is handled by CombinedLoss.
        # For standalone DiceLoss, user must ensure input is probabilities.
        # This example assumes if DiceLoss is primary, inputs are probs or handled by CombinedLoss logic.
        if model_outputs_logits:
            print(
                "Warning: Standalone DiceLoss expects probabilities. Model outputs logits. Sigmoid will be applied if CombinedLoss is not used.")
        criterion = DiceLoss(**CONFIG["loss_params"].get("DiceLoss", {})).to(device)
        # If standalone DiceLoss is used with a model outputting logits, you'd wrap it:
        # criterion = lambda pred_logits, target: DiceLoss(...)(torch.sigmoid(pred_logits), target)

    elif loss_name == "CombinedLoss":  # Handles (Focal or BCE variant) + Dice
        combined_loss_cfg = CONFIG["loss_params"].get("CombinedLoss", {})
        loss1_type_cfg = combined_loss_cfg.get("loss1_type", "FocalLoss")

        # Merge general params for loss1 with specific overrides from CombinedLoss config
        specific_loss1_params_base = CONFIG["loss_params"].get(loss1_type_cfg, {})
        combined_loss1_override_params = combined_loss_cfg.get("loss1_params", {})
        final_loss1_params = {**specific_loss1_params_base, **combined_loss1_override_params}

        # Merge general params for DiceLoss with specific overrides
        dice_params_base = CONFIG["loss_params"].get("DiceLoss", {})
        combined_dice_override_params = combined_loss_cfg.get("dice_params", {})
        final_dice_params = {**dice_params_base, **combined_dice_override_params}

        criterion = CombinedLoss(
            loss1_type=loss1_type_cfg,
            loss1_weight=combined_loss_cfg.get("loss1_weight", 0.5),
            loss1_params=final_loss1_params,
            dice_weight=combined_loss_cfg.get("dice_weight", 0.5),
            dice_params=final_dice_params,
            model_outputs_logits=model_outputs_logits  # Pass whether model outputs logits
        ).to(device)

    elif loss_name == "FocalBCELoss":  # Handles Focal + (BCEWithLogits or BCELoss)
        focal_bce_cfg = CONFIG["loss_params"].get("FocalBCELoss", {})

        # Merge general FocalLoss params with specific overrides from FocalBCELoss config
        focal_params_general = CONFIG["loss_params"].get("FocalLoss", {})
        focal_params_specific = focal_bce_cfg.get("focal_params", {})
        final_focal_params = {**focal_params_general, **focal_params_specific}

        # Determine BCE component type and merge its params
        bce_loss_type_cfg = focal_bce_cfg.get("bce_loss_type", "BCEWithLogitsLoss")
        bce_params_general = CONFIG["loss_params"].get(bce_loss_type_cfg, {})
        bce_params_specific = focal_bce_cfg.get("bce_params", {})
        final_bce_params = {**bce_params_general, **bce_params_specific}

        try:
            criterion = FocalBCELoss(
                focal_weight=focal_bce_cfg.get("focal_weight", 0.5),
                focal_params=final_focal_params,
                bce_weight=focal_bce_cfg.get("bce_weight", 0.5),
                bce_loss_type=bce_loss_type_cfg,
                bce_params=final_bce_params,
                inputs_are_logits=model_outputs_logits  # Pass whether model outputs logits
            ).to(device)
        except ValueError as e:
            print(f"Error initializing FocalBCELoss: {e}")
            print("Please check your model's output_activation_str and FocalBCELoss configuration in config.json.")
            # Fallback or exit
            # For example, fallback to a default safe loss or raise the error to stop execution
            # For now, let's re-raise to make the configuration error explicit.
            raise e


    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    # Optimizer
    if CONFIG["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    elif CONFIG["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    else:  # Default to Adam
        print(f"Optimizer {CONFIG['optimizer']} not fully supported, defaulting to Adam.")
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    print(f"Using optimizer: {CONFIG['optimizer']}")

    # Scheduler
    scheduler = None
    scheduler_cfg_params = CONFIG["scheduler_params"].get(CONFIG["scheduler"], {})
    if CONFIG["scheduler"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg_params)
    elif CONFIG["scheduler"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_cfg_params)
    elif CONFIG["scheduler"] == "CosineAnnealingLR":
        T_max_epochs = scheduler_cfg_params.get("T_max_epochs", CONFIG["num_epochs"])
        eta_min_factor = scheduler_cfg_params.get("eta_min_factor", 0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max_epochs,
                                                         eta_min=CONFIG["learning_rate"] * eta_min_factor)
    if scheduler: print(f"Using LR scheduler: {CONFIG['scheduler']}")

    # Training Loop
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    patience_counter = 0
    start_training_time = time.time()

    for epoch in range(CONFIG["num_epochs"]):
        epoch_start_time = time.time()
        if hasattr(train_dataset, 'reshuffle_indices'):
            train_dataset.reshuffle_indices()
        if hasattr(val_dataset, 'reshuffle_indices'):  # If val also uses subsetting
            val_dataset.reshuffle_indices()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, len(train_loader))
        val_loss = evaluate_model(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1:03d}/{CONFIG['num_epochs']:03d} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_duration:.2f}s"
        )

        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/validation_epoch', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif CONFIG["scheduler"] in ["CosineAnnealingLR",
                                         "StepLR"]:  # StepLR steps based on its internal counter usually
                scheduler.step()

        if (epoch + 1) % CONFIG["log_images_every_n_epochs"] == 0 or epoch == CONFIG["num_epochs"] - 1 or epoch == 0:
            visualize_predictions_with_emg_fit(model, device, CONFIG, epoch + 1, writer)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, f"best_model_epoch{epoch + 1}_val{val_loss:.4f}.pth"))
            print(f"  => Best model saved (Val Loss: {best_val_loss:.5f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if CONFIG.get("early_stopping_patience", float('inf')) <= patience_counter:
            print(
                f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss for {patience_counter} epochs.")
            break

    # End of Training
    total_training_time = time.time() - start_training_time
    print(f"\n--- Training Finished ---")
    print(
        f"Total training time: {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss:.5f}")

    # Save the final model (could be the same as best if early stopping didn't trigger late)
    torch.save(model.state_dict(), os.path.join(model_save_dir, "final_model.pth"))
    print(f"Final model saved to: {os.path.join(model_save_dir, 'final_model.pth')}")

    print("\nVisualizing predictions with the final model (including EMG fit attempt)...")
    # Load best model for final visualization if desired
    # model.load_state_dict(torch.load(os.path.join(model_save_dir, f"best_model...pth"))) # Find best model path
    # model.eval()
    visualize_predictions_with_emg_fit(model, device, CONFIG, "final", writer)

    writer.close()
    print(f"\nTensorBoard logs saved. Run: tensorboard --logdir \"{CONFIG['log_dir_base']}\"")


if __name__ == "__main__":
    main()
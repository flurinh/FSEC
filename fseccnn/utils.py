# fseccnn/utils.py

import json
import os
import logging
import re  # For parsing epoch and validation loss from filenames
import pandas as pd
import numpy as np
import torch  # For type hinting and potentially simple model ops if any

# --- Logging Setup (Basic) ---
# Configure logging if not already configured by the main application
# This is a simple setup; a more robust app might configure logging centrally.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Handling ---

DEFAULT_CONFIG_FILENAME = "config.json"


# Model filename will be determined dynamically by `find_best_model_path`

def load_config(config_path_or_dir):
    """
    Loads a JSON configuration file.
    If a directory is provided, it looks for 'config.json' inside it.
    """
    if os.path.isdir(config_path_or_dir):
        config_path = os.path.join(config_path_or_dir, DEFAULT_CONFIG_FILENAME)
    else:
        config_path = config_path_or_dir

    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {config_path}: {e}")
        raise ValueError(f"Invalid JSON format in {config_path}") from e
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise


def save_config(config_dict, filepath):
    """Saves a configuration dictionary to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        logging.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving configuration to {filepath}: {e}")
        raise


def find_best_model_path(model_folder_path):
    """
    Finds the 'best' model .pth file in the given folder.
    The 'best' model is assumed to be named like 'best_model_epochX_valY.pth'
    and the one with the highest epoch number (if multiple 'best_model_' files exist)
    or lowest validation loss for the same highest epoch is chosen.
    If no 'best_model_' prefix, it looks for 'final_model.pth'.
    If none of those, it takes the .pth file with 'best' in its name and highest epoch,
    or just any .pth file with the highest epoch if 'best' isn't present.

    Args:
        model_folder_path (str): Path to the directory containing model files.

    Returns:
        str or None: Path to the best model file, or None if not found.
    """
    if not os.path.isdir(model_folder_path):
        logging.error(f"Model folder not found: {model_folder_path}")
        return None

    pth_files = [f for f in os.listdir(model_folder_path) if f.endswith('.pth')]
    if not pth_files:
        logging.warning(f"No .pth files found in {model_folder_path}")
        return None

    best_models = []  # (epoch, val_loss, filepath)

    # Regex to parse epoch and val_loss from filenames like:
    # best_model_epochNUMBER_valFLOAT.pth
    # model_epochNUMBER_valFLOAT.pth
    # model_epochNUMBER.pth
    epoch_val_pattern = re.compile(r"epoch(\d+)(?:_val([\d\.]+))?\.pth$")

    for f in pth_files:
        match = epoch_val_pattern.search(f)
        epoch = -1
        val_loss = float('inf')  # Higher is worse

        if match:
            epoch = int(match.group(1))
            if match.group(2):  # If val_loss is present
                val_loss = float(match.group(2))
            else:  # If only epoch is present, assign a neutral val_loss for sorting
                val_loss = 0  # Or some other indicator if no val loss

        if f.startswith("best_model_"):
            best_models.append({'epoch': epoch, 'val_loss': val_loss, 'path': os.path.join(model_folder_path, f),
                                'type': 'best_prefix'})
        elif f == "final_model.pth":
            best_models.append({'epoch': epoch if epoch != -1 else float('inf'), 'val_loss': val_loss,
                                'path': os.path.join(model_folder_path, f),
                                'type': 'final'})  # Prioritize higher epoch for final
        elif "best" in f.lower():
            best_models.append({'epoch': epoch, 'val_loss': val_loss, 'path': os.path.join(model_folder_path, f),
                                'type': 'best_in_name'})
        else:
            best_models.append(
                {'epoch': epoch, 'val_loss': val_loss, 'path': os.path.join(model_folder_path, f), 'type': 'other'})

    if not best_models:
        logging.warning(f"No suitable model files identified in {model_folder_path}")
        return None

    # Sort models:
    # 1. 'best_model_' prefix files first
    # 2. 'final_model.pth' next
    # 3. Files with 'best' in their name
    # 4. Other .pth files
    # Within these categories, sort by highest epoch, then by lowest validation loss.
    def sort_key(m):
        type_priority = {'best_prefix': 0, 'final': 1, 'best_in_name': 2, 'other': 3}
        return (type_priority[m['type']], -m['epoch'], m['val_loss'])  # -epoch for descending epoch

    best_models.sort(key=sort_key)

    chosen_model = best_models[0]
    logging.info(
        f"Selected model: {chosen_model['path']} (Epoch: {chosen_model['epoch']}, Val Loss: {chosen_model['val_loss']})")
    return chosen_model['path']


def get_model_and_config_paths(model_folder_path):
    """
    Given a folder, finds the best model and config file paths.
    Returns (model_path, config_path) or (None, None) if not found.
    """
    best_model_filepath = find_best_model_path(model_folder_path)
    config_path = os.path.join(model_folder_path, DEFAULT_CONFIG_FILENAME)

    if not os.path.isfile(config_path):
        logging.warning(f"Config file {DEFAULT_CONFIG_FILENAME} not found in {model_folder_path}")
        config_path = None

    if best_model_filepath is None:
        logging.warning(f"No suitable model found in {model_folder_path} by find_best_model_path.")

    return best_model_filepath, config_path


# --- Data File Handling (FSEC CSV) ---

def load_fsec_csv(filepath, timepoint_col_name, sample_column_names=None, header_row=0):
    """
    Loads FSEC data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        timepoint_col_name (str): Name of the column containing timepoints/elution volume.
        sample_column_names (list of str, optional): Specific sample columns to load.
                                                     If None, attempts to load all numeric columns
                                                     excluding the timepoint column.
        header_row (int): Row number to use as column names (0-indexed).

    Returns:
        tuple: (dict_of_samples, list_of_all_numeric_cols_found)
               dict_of_samples: Keys are sample column names, values are
                                tuples of (numpy_array_x, numpy_array_y).
               Returns (None, None) if loading fails.
    """
    try:
        df = pd.read_csv(filepath, header=header_row)
        logging.info(f"Successfully read CSV: {filepath}")
    except FileNotFoundError:
        logging.error(f"CSV file not found: {filepath}")
        return None, None
    except Exception as e:
        logging.error(f"Error reading CSV file {filepath}: {e}")
        return None, None

    if timepoint_col_name not in df.columns:
        logging.error(
            f"Timepoint column '{timepoint_col_name}' not found in CSV. Available columns: {df.columns.tolist()}")
        return None, None

    # Ensure timepoint column is numeric and handle NaNs
    try:
        df[timepoint_col_name] = pd.to_numeric(df[timepoint_col_name], errors='coerce')
        if df[timepoint_col_name].isnull().any():
            logging.warning(
                f"NaN values found in timepoint column '{timepoint_col_name}'. Rows with NaN timepoints will be dropped.")
            df.dropna(subset=[timepoint_col_name], inplace=True)
        original_x_values = df[timepoint_col_name].to_numpy(dtype=float)
    except Exception as e:
        logging.error(f"Could not convert timepoint column '{timepoint_col_name}' to numeric: {e}")
        return None, None

    if len(original_x_values) == 0:
        logging.error(f"No valid timepoints found after processing column '{timepoint_col_name}'.")
        return None, None

    # Identify potential sample columns
    all_numeric_cols = []
    if sample_column_names is None:
        potential_sample_cols = [col for col in df.columns if col != timepoint_col_name]
        # Try to convert to numeric and see which ones work
        for col in potential_sample_cols:
            try:
                pd.to_numeric(df[col], errors='raise')  # Test conversion
                all_numeric_cols.append(col)
            except (ValueError, TypeError):
                logging.info(f"Column '{col}' is not purely numeric and will be skipped as a sample column.")
        if not all_numeric_cols:
            logging.error("No numeric sample columns found in the CSV besides the timepoint column.")
            return None, None
        sample_column_names_to_load = all_numeric_cols
    else:
        # Validate provided sample_column_names
        sample_column_names_to_load = []
        for col_name in sample_column_names:
            if col_name not in df.columns:
                logging.warning(f"Specified sample column '{col_name}' not found in CSV. Skipping.")
            else:
                sample_column_names_to_load.append(col_name)
        if not sample_column_names_to_load:
            logging.error("None of the specified sample columns were found in the CSV.")
            return None, None
        all_numeric_cols = sample_column_names_to_load  # In this case, user specified them

    output_data = {}
    for sample_col in sample_column_names_to_load:
        try:
            # Convert to numeric, coercing errors to NaN, then convert to float array
            y_values_series = pd.to_numeric(df[sample_col], errors='coerce')

            # Create a mask for valid (non-NaN) timepoints and corresponding y-values
            valid_mask = ~y_values_series.isnull()

            current_x = original_x_values[valid_mask]
            current_y = y_values_series[valid_mask].to_numpy(dtype=float)

            if len(current_x) == 0:  # Skip if all values were NaN for this sample
                logging.warning(
                    f"Sample column '{sample_col}' contains no valid numeric data after NaN removal. Skipping.")
                if sample_col in all_numeric_cols: all_numeric_cols.remove(sample_col)  # remove if it became invalid
                continue

            # Check for consistent X ordering if not already sorted (important for some processing)
            if not np.all(np.diff(current_x) >= 0):
                logging.warning(f"X-values for sample '{sample_col}' are not sorted. Sorting them now.")
                sort_indices = np.argsort(current_x)
                current_x = current_x[sort_indices]
                current_y = current_y[sort_indices]

            output_data[sample_col] = (current_x, current_y)
            logging.info(f"Loaded sample: '{sample_col}' with {len(current_y)} data points.")

        except Exception as e:
            logging.error(f"Error processing sample column '{sample_col}': {e}")
            if sample_col in all_numeric_cols: all_numeric_cols.remove(sample_col)

    if not output_data:
        logging.error("No valid sample data could be loaded from the CSV.")
        return None, None

    return output_data, all_numeric_cols


def export_results_to_csv(filepath, results_data_list_of_dicts, column_order=None):
    """
    Exports a list of dictionaries (each dict is a row) to a CSV file.

    Args:
        filepath (str): Path to save the CSV file.
        results_data_list_of_dicts (list): List of dictionaries, where keys are column headers.
        column_order (list, optional): Specific order for columns. If None, uses dict keys order.
    """
    if not results_data_list_of_dicts:
        logging.warning("No data provided to export_results_to_csv.")
        return

    try:
        df = pd.DataFrame(results_data_list_of_dicts)
        if column_order:
            # Ensure all requested columns exist, add missing ones with NaN if necessary
            for col in column_order:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[column_order]  # Reorder and select

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logging.info(f"Results successfully exported to {filepath}")
    except Exception as e:
        logging.error(f"Error exporting results to CSV {filepath}: {e}")
        raise


# --- PyTorch Model Related Utilities (Simple ones) ---

def get_device():
    """Returns the available PyTorch device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU.")
    return device


def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    if model is None: return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Miscellaneous Utilities ---

def ensure_list(item):
    """Ensures the item is a list. If not, wraps it in a list."""
    if item is None:
        return []
    if isinstance(item, list):
        return item
    return [item]


def format_peak_label(peak_params_dict, include_auc=False, precision=2):
    """
    Creates a formatted string label for a peak given its parameters.
    Example: peak_params_dict = {'mu': 10.234, 'sigma': 0.567, 'tau': 0.123, 'Rel_AUC_Percent': 95.5}
    """
    mu = peak_params_dict.get('Elution_Time_μ', peak_params_dict.get('mu'))
    sigma = peak_params_dict.get('Peak_Width_σ', peak_params_dict.get('sigma'))
    tau = peak_params_dict.get('Tailing_Factor_τ', peak_params_dict.get('tau'))

    label_parts = []
    if mu is not None:
        label_parts.append(f"μ:{mu:.{precision}f}")
    if sigma is not None:
        label_parts.append(f"σ:{sigma:.{precision}f}")
    if tau is not None and tau > 1e-3:  # Only show tau if significant
        label_parts.append(f"τ:{tau:.{precision}f}")

    if include_auc:
        auc_percent = peak_params_dict.get('Relative_AUC_Percent')
        if auc_percent is not None:
            label_parts.append(f"AUC:{auc_percent:.1f}%")

    return ", ".join(label_parts)


def get_unique_filepath(base_path, filename):
    """
    Generates a unique filepath by appending a counter if the file already exists.
    Example: if 'results.csv' exists, it tries 'results_1.csv', 'results_2.csv', etc.
    """
    full_path = os.path.join(base_path, filename)
    if not os.path.exists(full_path):
        return full_path

    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_full_path = os.path.join(base_path, new_filename)
        if not os.path.exists(new_full_path):
            return new_full_path
        counter += 1


def safe_filename(name_str):  # Renamed from safe_filename_gui for generic use
    """
    Sanitizes a string to be used as a safe filename component.
    Removes special characters, replaces spaces/hyphens with single hyphens.
    """
    if not isinstance(name_str, str):
        name_str = str(name_str)

    # Remove characters that are generally problematic in filenames across OS
    # Keep alphanumeric, underscore, hyphen, period.
    name_str = re.sub(r'[^\w\s.-]', '', name_str).strip()
    # Replace one or more spaces or hyphens with a single hyphen
    name_str = re.sub(r'[-\s]+', '-', name_str)
    # Remove leading/trailing hyphens that might result
    name_str = name_str.strip('-')

    return name_str if name_str else "unnamed_file_component"


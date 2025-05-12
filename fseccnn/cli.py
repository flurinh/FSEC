# cli.py
import argparse

def parse_predict_args():
    parser = argparse.ArgumentParser(description="FSEC Peak Detection and Deconvolution Script")
    # Changed to optional arguments with defaults
    parser.add_argument("--input_csv", type=str, default='input/xresults.csv',
                        help="Path to the input CSV file. Default: 'input/xresults.csv'")
    parser.add_argument("--model_path", type=str,
                        default='trained_fsec_models_offline/FSEC_UNet1D_YCorrBaseline_v1_20250511-182852/best_model_epoch10_val0.1259.pth', # Make sure path is valid
                        help="Path to the trained PyTorch model file. (Default: a placeholder path)")
    parser.add_argument("--config_path", type=str,
                        default='trained_fsec_models_offline/FSEC_UNet1D_YCorrBaseline_v1_20250511-182852/config.json', # Make sure path is valid
                        help="Path to the JSON config file. (Default: a placeholder path)")

    # Data loading arguments
    parser.add_argument("--timepoint_col_name", type=str, default="Timepoint",
                        help="Name of the column for X-axis data. Default: 'Timepoint'")
    # ... rest of your optional arguments (these are already correctly defined as optional flags) ...
    parser.add_argument("--sample_column_names", type=str, nargs='+', default=None,
                        help="List of sample column names to process. If not provided, all columns (except timepoint) are processed.")
    parser.add_argument("--delimiter", type=str, default=",", help="Delimiter used in the CSV file. Default: ','")
    parser.add_argument("--skip_rows", type=int, default=0, help="Number of rows to skip at the beginning. Default: 0")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Directory to save output plots and data. Default: 'predictions'")
    parser.add_argument("--save_plot", action="store_true",
                        help="Save the prediction and deconvolution plot for each sample.")
    parser.add_argument("--save_results_csv", action="store_true",
                        help="Save detected peaks and fitted parameters to CSV files.")
    parser.add_argument("--cnn_threshold", type=float, default=None,
                        help="Peak detection threshold for CNN output (0-1). Overrides config if set.")
    parser.add_argument("--cnn_min_distance", type=int, default=None,
                        help="Minimum distance (samples) between peaks for CNN. Overrides config if set.")
    parser.add_argument("--fitter_maxfev", type=int, default=None,
                        help="Max function evaluations for EMG fitter. Overrides fitter default/config.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run inference on ('auto', 'cpu', 'cuda'). Default: 'auto'")
    parser.add_argument("--no_emg_fit", action="store_true",
                        help="Skip the EMG fitting stage (only run CNN peak detection).")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_predict_args() # This will now use defaults if no args are given
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
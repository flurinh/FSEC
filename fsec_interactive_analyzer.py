# fsec_interactive_analyzer.py

import panel as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy # Only needed for savgol_filter if called directly (not needed here)

# --- Import your FSEC utilities ---
try:
    # Import the new simplified analysis function
    from fsec_utils import (
        load_fsec_data_combined,
        analyze_chromatogram_simplified, # <--- Use the new simplified function
        normalize_signal # Might need this to estimate ranges for sliders
    )
except ImportError as e:
    print(f"Error: Could not import from fsec_utils.py: {e}")
    print("Ensure fsec_utils.py is updated and in the same directory or Python path.")
    exit()

# --- Configuration ---
DATA_DIRECTORY = 'input'
# --- Default ABSOLUTE thresholds (adjust as needed for typical data) ---
DEFAULT_PEAK_PROMINENCE_ABS = 5.0 # Units of corrected signal
DEFAULT_PEAK_WIDTH = 10
DEFAULT_SHOULDER_DERIV_THRESH_ABS = 0.01 # Units of -2nd derivative
DEFAULT_MIN_FEATURE_HEIGHT_ABS = 1.0 # Units of corrected signal

# --- Load Data ---
try:
    print(f"Loading data from '{DATA_DIRECTORY}'...")
    combined_df = load_fsec_data_combined(directory=DATA_DIRECTORY)
    if combined_df.empty: raise ValueError(f"No data loaded from '{DATA_DIRECTORY}'.")
    if 'Timepoint' not in combined_df.columns: raise ValueError("'Timepoint' column missing.")
    print(f"Data loaded successfully. Shape: {combined_df.shape}")
    time_col = 'Timepoint' # Assuming 'Timepoint'
    signal_columns = [col for col in combined_df.columns if col != time_col and pd.api.types.is_numeric_dtype(combined_df[col])]
    if not signal_columns: raise ValueError("No numeric signal columns found (excluding time column).")
except Exception as e:
    print(f"Error during data loading: {e}. Exiting.")
    exit()

# --- Setup Panel ---
pn.extension(sizing_mode='stretch_width')

# --- Helper Function to Estimate Ranges (Optional but helpful) ---
# We run baseline correction once initially to estimate ranges for sliders
print("Estimating initial signal/derivative ranges for sliders...")
initial_col = signal_columns[0]
initial_signal = combined_df[initial_col].values
initial_time = combined_df[time_col].values
initial_corrected, initial_baseline, _, initial_win = normalize_signal(initial_signal, initial_time)
initial_max_corrected = np.nanmax(initial_corrected) if np.any(np.isfinite(initial_corrected)) else 100.0
initial_max_corrected = max(initial_max_corrected, 1.0) # Ensure range is at least 1

# Estimate derivative range (approximate)
initial_deriv_range = 0.1 # Default guess
if initial_win < len(initial_corrected):
    try:
        initial_s_for_deriv = np.copy(initial_corrected); nan_mask_i = np.isnan(initial_s_for_deriv)
        if np.any(nan_mask_i): x_i = np.arange(len(initial_s_for_deriv)); finite_i = ~nan_mask_i
        if np.sum(finite_i) >= 2: initial_s_for_deriv[nan_mask_i] = np.interp(x_i[nan_mask_i], x_i[finite_i], initial_s_for_deriv[finite_i]);
        if np.isnan(initial_s_for_deriv).any(): initial_s_for_deriv = pd.Series(initial_s_for_deriv).fillna(method='ffill').fillna(method='bfill').values
        else: initial_s_for_deriv[nan_mask_i] = 0.0
        initial_d2 = scipy.signal.savgol_filter(initial_s_for_deriv, initial_win, 2, deriv=2)
        initial_neg_d2 = -initial_d2[np.isfinite(initial_d2)]
        if len(initial_neg_d2) > 0:
             initial_deriv_range = max(np.max(initial_neg_d2), 0.01) # Use max as upper bound guide
    except Exception as e: print(f"Warning: Could not estimate initial derivative range: {e}")

print(f"Initial estimates: Max Corrected Signal ~ {initial_max_corrected:.2f}, Max Neg. 2nd Deriv ~ {initial_deriv_range:.4f}")

# --- Create Widgets (Updated for Absolute Thresholds) ---
column_select = pn.widgets.Select(
    name='Select Signal Column', options=signal_columns, value=initial_col)

# Absolute Prominence Slider
peak_prominence_slider = pn.widgets.FloatSlider(
    name='Peak Prominence (Absolute Units)',
    start=0.1, end=initial_max_corrected * 0.5, step=0.1, # Range up to 50% of initial max corrected signal
    value=min(DEFAULT_PEAK_PROMINENCE_ABS, initial_max_corrected * 0.5) # Ensure default fits range
)

width_slider = pn.widgets.IntSlider(
    name='Peak Width (Points)', start=3, end=100, step=1, value=DEFAULT_PEAK_WIDTH)

# Absolute Shoulder Derivative Threshold Slider
shoulder_deriv_thresh_slider = pn.widgets.FloatSlider(
    name='Shoulder Deriv Threshold (-d2 height)',
    start=0.0, end=initial_deriv_range * 1.5, step=initial_deriv_range / 100, # Range based on initial estimate
    value=min(DEFAULT_SHOULDER_DERIV_THRESH_ABS, initial_deriv_range * 1.5), # Ensure default fits range
    format='0[.]0000'
)

# Absolute Minimum Feature Height Slider
min_feature_height_slider = pn.widgets.FloatSlider(
    name='Min Feature Height (Absolute Units)',
    start=0.0, end=initial_max_corrected * 0.25, step=0.1, # Range up to 25% of initial max corrected signal
    value=min(DEFAULT_MIN_FEATURE_HEIGHT_ABS, initial_max_corrected * 0.25)
)

debug_check = pn.widgets.Checkbox( # Renamed for clarity
    name='Show Debug Info (in console)', value=False)

# Text hints for ranges
hint_text = pn.pane.Markdown(f"""
*Hints for **{initial_col}**:*
*Max corrected signal height: ~{initial_max_corrected:.2f}*
*Max neg. 2nd deriv value: ~{initial_deriv_range:.4f}*
*(Ranges based on initial load, adjust sliders as needed)*
""")

# --- Define Reactive Plotting Function (Updated) ---
@pn.depends(
    column_select.param.value,
    peak_prominence_slider.param.value, # Watch new slider
    width_slider.param.value,
    shoulder_deriv_thresh_slider.param.value, # Watch new slider
    min_feature_height_slider.param.value, # Watch new slider
    debug_check.param.value, # Watch renamed widget
    watch=True
)
def create_analysis_plots_simplified(column_name, peak_prominence, peak_width,
                                     shoulder_deriv_threshold, min_feature_height, debug):
    """
    Calls analyze_chromatogram_simplified and generates the Matplotlib plots.
    Returns a Matplotlib Figure object.
    """
    print(f"\n--- Re-analyzing: {column_name} (Simplified) ---")
    print(f"Params: PeakProm={peak_prominence:.4f}, PeakWidth={peak_width}, "
          f"ShoulderDerivThresh={shoulder_deriv_threshold:.5f}, MinHeight={min_feature_height:.4f}")

    # Call the NEW simplified analysis function
    analysis_results = analyze_chromatogram_simplified(
        combined_df,
        column_name=column_name,
        peak_prominence=peak_prominence, # Pass absolute value
        peak_width=peak_width,
        shoulder_deriv_threshold=shoulder_deriv_threshold, # Pass absolute value
        min_feature_height=min_feature_height, # Pass absolute value
        plot=False, # We handle plotting here
        debug=debug
    )

    # --- Create Matplotlib Plots ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2 = axes

    if analysis_results:
        # Data extracted from results
        timepoints = analysis_results['timepoints']
        signal = analysis_results['original_signal']
        baseline = analysis_results['baseline']
        corrected_signal = analysis_results['corrected_signal']
        # Note: normalized_corrected_signal is not returned by simplified func
        peak_indices = analysis_results['peak_indices'] # These are FINAL peaks
        shoulder_indices = analysis_results['shoulder_indices'] # These are FINAL shoulders
        sg_window_size = analysis_results['sg_window_size']
        params_used = analysis_results['params'] # Get params used

        # --- Plotting Logic ---
        ax1.plot(timepoints, signal, label='Original Signal', color='gray', alpha=0.6, lw=1.5)
        ax1.plot(timepoints, corrected_signal, label='Corrected Signal', color='blue', lw=1.5)
        ax1.plot(timepoints, baseline, label=f'Baseline (SNIP, win={sg_window_size})', color='orange', linestyle='--', lw=1.5)
        ax1.axhline(params_used['min_feature_height'], color='red', linestyle=':', alpha=0.5, lw=1, label=f'Min Height ({params_used["min_feature_height"]:.2f})')

        n_peaks = len(peak_indices); n_shoulders = len(shoulder_indices)
        if n_peaks > 0: ax1.plot(timepoints[peak_indices], corrected_signal[peak_indices], "x", color='red', markersize=10, mew=2, label=f'Peaks ({n_peaks})', linestyle='None')
        else: ax1.plot([],[], "x", color='red', markersize=10, mew=2, label='Peaks (0)')
        if n_shoulders > 0: ax1.plot(timepoints[shoulder_indices], corrected_signal[shoulder_indices], "o", color='green', markersize=8, label=f'Shoulders ({n_shoulders})', alpha=0.9, markerfacecolor='none', mew=2, linestyle='None')
        else: ax1.plot([],[], "o", color='green', markersize=8, markerfacecolor='none', mew=2, label='Shoulders (0)')

        ax1.set_title(f'Interactive Simplified Analysis ({column_name})'); ax1.set_ylabel('Signal')
        ax1.legend(loc='upper right', fontsize='small'); ax1.grid(True, alpha=0.4, linestyle=':')
        finite_y = np.concatenate([s[np.isfinite(s)] for s in [signal, corrected_signal, baseline] if s is not None])
        if len(finite_y)>0: y_min, y_max = np.min(finite_y), np.max(finite_y); y_buffer = (y_max - y_min) * 0.05 if (y_max - y_min)>0 else 0.1; ax1.set_ylim(bottom=min(y_min - y_buffer, -y_buffer), top=y_max + y_buffer)

        # Bottom plot: Negative Second derivative
        deriv_window = max(5, sg_window_size if sg_window_size % 2 != 0 else sg_window_size + 1)
        if deriv_window < len(corrected_signal):
            signal_for_deriv = np.copy(corrected_signal); nan_mask_deriv = np.isnan(signal_for_deriv)
            if np.any(nan_mask_deriv): # Re-interpolate locally just for plotting deriv
                 x_coords_d = np.arange(len(signal_for_deriv)); finite_mask_d = ~nan_mask_deriv
                 if np.sum(finite_mask_d) >= 2: signal_for_deriv[nan_mask_deriv] = np.interp(x_coords_d[nan_mask_deriv], x_coords_d[finite_mask_d], signal_for_deriv[finite_mask_d]);
                 if np.isnan(signal_for_deriv).any(): signal_for_deriv = pd.Series(signal_for_deriv).fillna(method='ffill').fillna(method='bfill').values
                 else: signal_for_deriv[nan_mask_deriv] = 0.0
            try:
                second_derivative = scipy.signal.savgol_filter(signal_for_deriv, deriv_window, 2, deriv=2)
                neg_second_derivative = -second_derivative
                neg_second_derivative[~np.isfinite(neg_second_derivative)] = 0.0
                ax2.plot(timepoints, neg_second_derivative, label='-2nd Deriv', color='purple', lw=1.5)
                # Show the ABSOLUTE derivative threshold used
                ax2.axhline(params_used['shoulder_deriv_threshold'], color='magenta', linestyle=':', alpha=0.7, lw=1, label=f'Deriv Thresh ({params_used["shoulder_deriv_threshold"]:.3f})')
            except Exception as e: print(f"Error plotting derivative: {e}"); ax2.text(0.5, 0.5, "Error plotting derivative", ha='center', va='center', transform=ax2.transAxes)
        else: ax2.text(0.5, 0.5, "Signal too short for 2nd derivative plot", ha='center', va='center', transform=ax2.transAxes)
        time_col_name = combined_df.columns[0]
        ax2.set_xlabel(f'{time_col_name}'); ax2.set_ylabel('Neg. 2nd Deriv'); ax2.axhline(0, color='black', linewidth=0.5, linestyle='--'); ax2.grid(True, alpha=0.4, linestyle=':')
        ax2.legend(loc='upper right', fontsize='small'); ax2.autoscale(enable=True, axis='y', tight=True);

    else:
        ax1.text(0.5, 0.5, f"Analysis failed for {column_name}.\nCheck console.", ha='center', va='center', color='red'); ax1.set_title(f'Analysis Failed ({column_name})')
        ax2.text(0.5, 0.5, "No derivative data", ha='center', va='center')

    plt.tight_layout(pad=0.5)
    plt.close(fig) # Close plot to prevent display outside panel
    return fig

# --- Create Panel Layout (Updated Widgets) ---
widgets_column = pn.Column(
    pn.pane.Markdown("### Analysis Parameters (Simplified)"),
    column_select,
    peak_prominence_slider, # Absolute
    width_slider,
    shoulder_deriv_thresh_slider, # Absolute
    min_feature_height_slider, # Absolute
    debug_check, # Renamed
    hint_text, # Add hints
    width=300 # Wider sidebar for new sliders
)

# Use the new reactive function
plot_pane = pn.pane.Matplotlib(create_analysis_plots_simplified, dpi=100)

dashboard = pn.Row(
    widgets_column,
    plot_pane,
    sizing_mode='stretch_width'
)

# --- Make it servable ---
dashboard.servable(title="FSEC Interactive Analyzer (Simplified)")
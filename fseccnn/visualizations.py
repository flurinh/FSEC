# fseccnn/visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import logging  # Added for potential logging within visualization functions

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())


# --- Core Visualization Functions ---

def plot_deconvolution_summary(
        original_x,
        original_y,
        fitted_total_y=None,
        individual_emg_components_data=None,
        fitted_baseline_y=None,
        cnn_prob_map_original_x=None,
        cnn_detected_peak_x_coords=None,
        title="FSEC Deconvolution",
        ax=None
):
    """
    Generates the main deconvolution plot.

    Args:
        original_x (np.array): X-values of the original chromatogram.
        original_y (np.array): Y-values of the original chromatogram.
        fitted_total_y (np.array, optional): Y-values of the sum of all fitted EMGs + baseline.
        individual_emg_components_data (list, optional):
            List of dicts for each component. Each dict should contain:
            'x_coords': np.array of x-values for this component's plot (usually original_x).
            'y_coords_on_baseline': np.array of y-values (EMG_shape_i + fitted_baseline_y).
            'label': str for legend.
        fitted_baseline_y (np.array, optional): Y-values of the fitted baseline.
        cnn_prob_map_original_x (np.array, optional): CNN probability map on original_x scale.
        cnn_detected_peak_x_coords (np.array, optional): X-coordinates of CNN detected peaks.
        title (str): Plot title.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    if not isinstance(original_x, np.ndarray) or not isinstance(original_y, np.ndarray) or len(original_x) == 0:
        logger.warning("plot_deconvolution_summary: original_x or original_y is not a valid numpy array or is empty.")
        fig_empty, ax_empty = plt.subplots(figsize=(12, 7))
        ax_empty.text(0.5, 0.5, "No data to display.", ha='center', va='center', transform=ax_empty.transAxes)
        ax_empty.set_title(title)
        return fig_empty, ax_empty

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(12, 7))
    else:
        ax1 = ax
        fig = ax1.get_figure()
        ax1.clear()  # Clear existing axes if reused

    # 1. Original Data
    ax1.plot(original_x, original_y, label="Raw Data", color="black", linewidth=1.5, zorder=10)

    # 2. CNN Detections (Vertical Lines)
    # Ensure cnn_detected_peak_x_coords is an iterable (like list or np.array) and not None
    if cnn_detected_peak_x_coords is not None and hasattr(cnn_detected_peak_x_coords, '__iter__') and len(
            cnn_detected_peak_x_coords) > 0:
        # Label only the first line to avoid legend clutter
        ax1.axvline(cnn_detected_peak_x_coords[0], color='deepskyblue', linestyle=':', linewidth=1.2, alpha=0.8,
                    zorder=8, label='CNN Detection')
        for i in range(1, len(cnn_detected_peak_x_coords)):
            ax1.axvline(cnn_detected_peak_x_coords[i], color='deepskyblue', linestyle=':', linewidth=1.2, alpha=0.8,
                        zorder=8, label="_nolegend_")

    # 3. Fitted Baseline
    if fitted_baseline_y is not None and len(fitted_baseline_y) == len(original_x):
        ax1.plot(original_x, fitted_baseline_y, label="Fitted Baseline", color="green", linestyle=":", linewidth=1.5,
                 zorder=5)

    # 4. Individual Fitted Components
    if individual_emg_components_data:  # Check if list is not None and not empty
        num_components = len(individual_emg_components_data)
        # Ensure colors are generated even if num_components is 1
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, num_components)))  # Use 0.1 to 0.9 to avoid extreme colors

        for i, comp_data in enumerate(individual_emg_components_data):
            current_x_coords = comp_data.get('x_coords', original_x)
            y_component_on_baseline = comp_data.get('y_coords_on_baseline')
            component_label = comp_data.get('label', f"Component {i + 1}")

            if y_component_on_baseline is not None and len(y_component_on_baseline) == len(current_x_coords):
                # Determine base for filling (either fitted_baseline_y or zeros)
                base_for_fill_values = np.zeros_like(current_x_coords)
                if fitted_baseline_y is not None and len(fitted_baseline_y) == len(current_x_coords):
                    base_for_fill_values = fitted_baseline_y

                ax1.fill_between(
                    current_x_coords, base_for_fill_values, y_component_on_baseline,
                    alpha=0.5, label=component_label,
                    color=colors[i % len(colors)], zorder=15)
                # Plot line for the component on top of baseline for better definition
                ax1.plot(current_x_coords, y_component_on_baseline, color=colors[i % len(colors)], linewidth=1.2,
                         zorder=16, label="_nolegend_")
            else:
                logger.warning(f"Skipping component {i + 1} in plot due to missing or mismatched y_coords_on_baseline.")

    # 5. Total Fitted Curve
    if fitted_total_y is not None and len(fitted_total_y) == len(original_x):
        ax1.plot(original_x, fitted_total_y, label="Total Fit", color="red", linestyle="--", linewidth=2, zorder=20)

    ax1.set_xlabel("Elution Time / Volume")
    ax1.set_ylabel("Signal Intensity")
    ax1.set_title(title)

    # Y-limit adjustment
    all_y_for_lims = [original_y]
    if fitted_total_y is not None and len(fitted_total_y) > 0: all_y_for_lims.append(fitted_total_y)
    if fitted_baseline_y is not None and len(fitted_baseline_y) > 0: all_y_for_lims.append(fitted_baseline_y)
    # Also consider individual components if they might exceed total fit (shouldn't happen if positive)
    # if individual_emg_components_data:
    #     for comp_data in individual_emg_components_data:
    #         if comp_data.get('y_coords_on_baseline') is not None:
    #             all_y_for_lims.append(comp_data.get('y_coords_on_baseline'))

    min_val_plot, max_val_plot = np.nanmin(original_y), np.nanmax(original_y)  # Start with original data
    for y_arr in all_y_for_lims:
        if y_arr is not None and len(y_arr) > 0:
            min_val_plot = min(min_val_plot, np.nanmin(y_arr))
            max_val_plot = max(max_val_plot, np.nanmax(y_arr))

    y_span = max_val_plot - min_val_plot if max_val_plot > min_val_plot else 1.0
    y_span = max(y_span, 1e-6)
    ax1.set_ylim(min_val_plot - 0.05 * y_span, max_val_plot + 0.05 * y_span)

    ax1.grid(True, linestyle='--', alpha=0.6)

    legend_handles, legend_labels = [], []
    h1_temp, l1_temp = ax1.get_legend_handles_labels()
    legend_handles.extend(h1_temp);
    legend_labels.extend(l1_temp)

    # Twin axis for CNN probability map
    if cnn_prob_map_original_x is not None and len(cnn_prob_map_original_x) == len(original_x):
        ax2 = ax1.twinx()
        ax2.plot(original_x, cnn_prob_map_original_x, label="CNN Peak Prob.", color="purple", alpha=0.35,
                 linestyle="-.", zorder=1)

        purple_rgba_label = (0.5, 0.0, 0.5, 0.6)
        purple_rgba_ticks = (0.5, 0.0, 0.5, 0.5)
        ax2.set_ylabel("CNN Peak Probability", color=purple_rgba_label)
        ax2.tick_params(axis='y', labelcolor=purple_rgba_ticks, colors=purple_rgba_ticks)

        max_prob_val = np.nanmax(cnn_prob_map_original_x) if len(cnn_prob_map_original_x) > 0 else 0
        ax2.set_ylim(-0.05, (1.1 * max_prob_val) if max_prob_val > 1e-3 else 0.1)

        ax2.spines["right"].set_edgecolor(purple_rgba_label[:3])
        ax2.spines["right"].set_alpha(purple_rgba_label[3])
        ax2.spines["right"].set_visible(True)

        h2_temp, l2_temp = ax2.get_legend_handles_labels()
        legend_handles.extend(h2_temp);
        legend_labels.extend(l2_temp)

    # Consolidate legends
    if legend_labels:
        # Filter out labels starting with "_nolegend_"
        filtered_handles_labels = [(h, l) for h, l in zip(legend_handles, legend_labels) if
                                   l and not l.startswith("_nolegend_")]
        if filtered_handles_labels:
            handles_final, labels_final = zip(*filtered_handles_labels)
            unique_legends = {label: handle for handle, label in
                              zip(handles_final, labels_final)}  # Keep last if duplicate labels

            ax1.legend(unique_legends.values(), unique_legends.keys(), loc='upper left',
                       bbox_to_anchor=(1.03, 1), borderaxespad=0., fontsize='small')
            # Adjust layout for external legend
            if fig: fig.tight_layout(rect=[0, 0, 0.80, 1])  # Leave more space for legend
        elif fig:  # No valid legend items
            fig.tight_layout()
    elif fig:  # No legend items at all
        fig.tight_layout()

    return fig, ax1


def plot_cnn_processing_view(
        cnn_input_x,
        cnn_input_y_normalized,
        cnn_output_prob_map,
        cnn_threshold_value=None,
        title="CNN Processing View",
        ax=None
):
    if not isinstance(cnn_input_x, np.ndarray) or not isinstance(cnn_input_y_normalized, np.ndarray) or \
            not isinstance(cnn_output_prob_map, np.ndarray) or \
            len(cnn_input_x) == 0 or len(cnn_input_y_normalized) == 0 or len(cnn_output_prob_map) == 0 or \
            len(cnn_input_x) != len(cnn_input_y_normalized) or len(cnn_input_x) != len(cnn_output_prob_map):
        logger.warning("plot_cnn_processing_view: Input arrays are invalid, empty, or mismatched in length.")
        fig_empty, ax_empty = plt.subplots(figsize=(10, 5))
        ax_empty.text(0.5, 0.5, "No valid CNN processing data to display.", ha='center', va='center',
                      transform=ax_empty.transAxes)
        ax_empty.set_title(title)
        return fig_empty, ax_empty

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 5))
    else:
        ax1 = ax
        fig = ax1.get_figure()
        ax1.clear()

    ax1.plot(cnn_input_x, cnn_input_y_normalized, label="Normalized Input to CNN", color="blue", alpha=0.8)
    ax1.set_xlabel("Canonical X-axis (CNN internal)")
    ax1.set_ylabel("Normalized Signal", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')

    ax2 = ax1.twinx()
    ax2.plot(cnn_input_x, cnn_output_prob_map, label="CNN Output Probability", color="darkorange", linestyle="-",
             alpha=0.8)
    if cnn_threshold_value is not None and len(cnn_input_x) > 0:  # Check if cnn_input_x is not empty
        ax2.hlines(cnn_threshold_value, cnn_input_x[0], cnn_input_x[-1],
                   color='red', linestyle=':', label=f'CNN Thresh ({cnn_threshold_value:.2f})', alpha=0.9)
    ax2.set_ylabel("CNN Peak Probability", color="darkorange")
    ax2.tick_params(axis='y', labelcolor="darkorange")

    max_prob_val = np.nanmax(cnn_output_prob_map) if len(cnn_output_prob_map) > 0 else 0
    ax2.set_ylim(-0.05, (1.05 * max_prob_val) if max_prob_val > 1e-3 else 0.1)
    ax2.grid(True, linestyle=':', alpha=0.5, axis='y')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    if all_labels:  # Only create legend if there are items
        unique_legends = {label: handle for handle, label in zip(all_handles, all_labels) if
                          label and not label.startswith("_nolegend_")}
        if unique_legends:
            ax1.legend(unique_legends.values(), unique_legends.keys(), loc='upper center',
                       bbox_to_anchor=(0.5, -0.12), ncol=min(3, len(unique_legends)), fontsize='small')  # Adjust ncol

    ax1.set_title(title)
    if fig: fig.tight_layout(rect=[0, 0.08, 1, 1])  # Adjust bottom margin more for legend
    return fig, ax1


def plot_raw_signals_overview(
        signals_dict,
        timepoint_col_name="Elution Time / Volume",
        title="Raw Signals Overview",
        max_signals_to_plot=15,  # Increased default slightly
        highlight_sample=None
):
    if not signals_dict:  # Handle empty signals_dict
        logger.warning("plot_raw_signals_overview: signals_dict is empty.")
        fig_empty, ax_empty = plt.subplots(figsize=(12, 7))
        ax_empty.text(0.5, 0.5, "No signals to display.", ha='center', va='center', transform=ax_empty.transAxes)
        ax_empty.set_title(title);
        ax_empty.set_xlabel(timepoint_col_name);
        ax_empty.set_ylabel("Signal Intensity")
        if fig_empty: fig_empty.tight_layout();
        return fig_empty, ax_empty

    fig, ax = plt.subplots(figsize=(12, 7))
    plotted_count = 0  # Count of signals actually added to legend

    # Sort items for consistent color mapping if max_signals_to_plot is less than total
    # or simply to have a defined order.
    sorted_sample_names = sorted(signals_dict.keys())

    items_to_plot_ordered = [(name, signals_dict[name]) for name in sorted_sample_names]

    # If highlight_sample exists, move it to the end to plot last (on top)
    highlight_item_tuple = None
    if highlight_sample and highlight_sample in signals_dict:
        highlight_item_tuple = (highlight_sample, signals_dict[highlight_sample])
        items_to_plot_ordered = [item for item in items_to_plot_ordered if item[0] != highlight_sample]
        # items_to_plot_ordered.append(highlight_item_tuple) # Will be plotted separately if needed

    num_to_color = min(len(items_to_plot_ordered), max_signals_to_plot)
    if highlight_sample and highlight_sample not in [item[0] for item in items_to_plot_ordered[:num_to_color]]:
        # Ensure highlight_sample gets a distinct color if it's outside the first `num_to_color`
        # This logic can be complex; simpler to plot highlighted one last with specific style.
        pass

    colors = plt.cm.viridis(np.linspace(0, 0.95, max(1, num_to_color)))  # Use 0-0.95 for better color range

    # Plot non-highlighted signals first
    for i, (sample_name, (x_raw, y_raw)) in enumerate(items_to_plot_ordered):
        if sample_name == highlight_sample:
            continue  # Skip highlighted for now, plot it last

        if plotted_count >= max_signals_to_plot:
            # Optionally plot remaining non-highlighted ones faintly
            # ax.plot(np.array(x_raw, dtype=float), np.array(y_raw, dtype=float), color='lightgrey', alpha=0.3, linewidth=0.5, zorder=0)
            continue

        y = np.array(y_raw, dtype=float)
        x = np.array(x_raw, dtype=float)
        nan_mask = np.isnan(y)

        current_color = colors[plotted_count % len(colors)] if len(colors) > 0 else 'grey'
        label_name = sample_name

        if np.any(nan_mask):
            y_interp = y.copy()
            x_non_nan, y_non_nan = x[~nan_mask], y[~nan_mask]
            if len(x_non_nan) >= 2:
                y_interp[nan_mask] = np.interp(x[nan_mask], x_non_nan, y_non_nan)
                ax.plot(x, y_interp, label=label_name, color=current_color, alpha=0.7, linewidth=1.0, zorder=5)
            else:
                ax.plot(x, y, label=f"{label_name} (NaNs)", color=current_color, alpha=0.7, linewidth=1.0, zorder=5)
        else:
            ax.plot(x, y, label=label_name, color=current_color, alpha=0.7, linewidth=1.0, zorder=5)
        plotted_count += 1

    # Plot highlighted sample last and more prominently
    if highlight_item_tuple:
        sample_name, (x_raw, y_raw) = highlight_item_tuple
        y = np.array(y_raw, dtype=float)
        x = np.array(x_raw, dtype=float)
        nan_mask = np.isnan(y)
        label_name = f"{sample_name} (Selected)"

        if np.any(nan_mask):
            y_interp = y.copy()
            x_non_nan, y_non_nan = x[~nan_mask], y[~nan_mask]
            if len(x_non_nan) >= 2:
                y_interp[nan_mask] = np.interp(x[nan_mask], x_non_nan, y_non_nan)
                ax.plot(x, y_interp, label=label_name, color='red', alpha=1.0, linewidth=1.8, zorder=10)
            else:
                ax.plot(x, y, label=f"{label_name} (NaNs)", color='red', alpha=1.0, linewidth=1.8, zorder=10)
        else:
            ax.plot(x, y, label=label_name, color='red', alpha=1.0, linewidth=1.8, zorder=10)
        if plotted_count < max_signals_to_plot: plotted_count += 1  # Count if it wasn't already counted

    if len(signals_dict) > plotted_count and plotted_count == max_signals_to_plot:
        ax.text(0.05, 0.05, f"...and {len(signals_dict) - plotted_count} more signals not shown.",
                transform=ax.transAxes, fontsize=9, color='grey',
                bbox=dict(boxstyle='round,pad=0.3', fc=(1, 1, 0.9), alpha=0.8))

    ax.set_xlabel(timepoint_col_name)
    ax.set_ylabel("Signal Intensity")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only show legend if there's something to label
        legend_ncol = max(1, min(4, len(handles) // 6 if len(handles) > 5 else 1))
        ax.legend(loc='best', fontsize='small', ncol=legend_ncol)

    ax.grid(True, linestyle='--', alpha=0.6)
    if fig: fig.tight_layout()
    return fig, ax


# --- Placeholder QC Plot Functions ---
# These remain placeholders as their implementation depends on
# data structures and logic not yet defined for batch QC.

def plot_batch_qc_grid(*args, **kwargs):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "Batch QC Grid - Not Implemented", ha='center', va='center', transform=ax.transAxes)
    logger.info("plot_batch_qc_grid called (placeholder).")
    return fig, [ax]


def plot_qc_metric_scatter(*args, **kwargs):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "QC Metric Scatter - Not Implemented", ha='center', va='center', transform=ax.transAxes)
    logger.info("plot_qc_metric_scatter called (placeholder).")
    return fig, ax


def plot_qc_metric_histogram(*args, **kwargs):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "QC Metric Histogram - Not Implemented", ha='center', va='center', transform=ax.transAxes)
    logger.info("plot_qc_metric_histogram called (placeholder).")
    return fig, ax
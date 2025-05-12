# gui.py (Phase 2 - Corrected Preprocessing Alignment)
import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import logging
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QLineEdit,
    QTextEdit, QSizePolicy, QGroupBox, QGridLayout, QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- Import custom project modules ---
try:
    # If gui.py is in project root, modules in fseccnn/
    from fseccnn import utils
    from fseccnn import visualizations
    from fseccnn.model import UNet1D
    # Assuming main_gui.py contains these helpers (or they are moved to a shared gui_utils.py)
    from fseccnn.main_gui import get_norm_activation_layers_gui, load_model_for_gui, preprocess_for_cnn_and_fitter
except ImportError:
    # If gui.py is inside fseccnn/
    try:
        import utils
        import visualizations
        from model import UNet1D
        from main_gui import get_norm_activation_layers_gui, load_model_for_gui, preprocess_for_cnn_and_fitter
    except ImportError as e_script:
        print(f"CRITICAL ERROR: Could not import custom modules: {e_script}")
        # Try to log if utils.logging is available
        if 'utils' in sys.modules and hasattr(utils, 'logging'):
            utils.logging.critical(f"Failed to import all custom modules for GUI: {e_script}", exc_info=True)
        sys.exit(1)


# --- Matplotlib Canvas Widget ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = plt.Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self._add_initial_text("Canvas Initialized")

    def _add_initial_text(self, text):
        try:
            if self.figure.axes:  # Check if figure has any axes
                self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, text, ha='center', va='center', color='grey', fontsize=10)
            self.draw()
        except Exception as e:
            print(f"Error in _add_initial_text: {e}")

    def update_plot(self, new_figure_obj):
        if new_figure_obj is None:
            self._add_initial_text("Plotting Error or No Data")
            if 'utils' in sys.modules and hasattr(utils, 'logging'):  # Check if logging is available
                utils.logging.warning("MplCanvas.update_plot received None for new_figure_obj.")
            return
        self.figure = new_figure_obj
        self.draw()

    def clear_canvas(self, message="Plot Cleared"):
        self._add_initial_text(message)


# --- Main Application Window ---
class FSECAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSEC Analyzer (Phase 2: CNN Peaks - Corrected Alignment)")
        self.setGeometry(50, 50, 1400, 800)

        self.csv_path = ""
        self.loaded_fsec_data = None
        self.all_identified_sample_cols = []
        self.timepoint_col_suggestion = "Timepoint"
        self.current_selected_sample_name = None
        self.current_sample_analysis_results = None

        self.model_folder_path = ""
        self.cnn_model = None
        self.train_config = None
        self.device = utils.get_device()

        # Parameters for CNN peak finding (can be made more configurable later)
        self.cnn_peak_threshold = 0.4
        self.cnn_min_peak_distance = 5  # In samples of original data, after prob map interpolation

        self.initUI()
        self.log_message(f"Application started. Using device: {self.device}. Load CSV and Model.", status=True)

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        app_layout = QHBoxLayout(main_widget)

        left_panel = QWidget();
        left_layout = QVBoxLayout(left_panel);
        left_panel.setMaximumWidth(400)

        file_group = QGroupBox("Load Data & Model");
        file_grid_layout = QGridLayout(file_group)
        self.btn_load_csv = QPushButton("1. Load FSEC CSV");
        self.btn_load_csv.clicked.connect(self.load_csv_file_triggered)
        self.lbl_csv_path = QLabel("CSV: Not loaded");
        self.lbl_csv_path.setWordWrap(True)
        self.btn_load_model_folder = QPushButton("2. Load Model Folder");
        self.btn_load_model_folder.clicked.connect(self.load_model_folder_triggered)
        self.lbl_model_path = QLabel("Model: Not loaded");
        self.lbl_model_path.setWordWrap(True)
        file_grid_layout.addWidget(self.btn_load_csv, 0, 0);
        file_grid_layout.addWidget(self.lbl_csv_path, 0, 1)
        file_grid_layout.addWidget(self.btn_load_model_folder, 1, 0);
        file_grid_layout.addWidget(self.lbl_model_path, 1, 1)
        file_grid_layout.addWidget(QLabel("Timepoint Col:"), 2, 0)
        self.txt_timepoint_col = QLineEdit(self.timepoint_col_suggestion);
        self.txt_timepoint_col.editingFinished.connect(self.update_data_after_timepoint_change)
        file_grid_layout.addWidget(self.txt_timepoint_col, 2, 1)
        left_layout.addWidget(file_group)

        sample_group = QGroupBox("Sample Selection & Analysis");
        sample_v_layout = QVBoxLayout(sample_group)
        sample_v_layout.addWidget(QLabel("Select Sample:"))
        self.combo_samples = QComboBox();
        self.combo_samples.currentTextChanged.connect(self.on_sample_selected_changed)
        sample_v_layout.addWidget(self.combo_samples)
        self.btn_analyze_sample = QPushButton("3. Analyze Selected (CNN Only)");
        self.btn_analyze_sample.clicked.connect(self.run_analysis_for_selected_sample_triggered);
        self.btn_analyze_sample.setEnabled(False)
        sample_v_layout.addWidget(self.btn_analyze_sample)
        self.btn_view_all_inputs = QPushButton("View All Loaded Raw Signals");
        self.btn_view_all_inputs.clicked.connect(self.plot_all_input_signals_triggered);
        self.btn_view_all_inputs.setEnabled(False)
        sample_v_layout.addWidget(self.btn_view_all_inputs)
        left_layout.addWidget(sample_group)

        param_group = QGroupBox("CNN Parameters (Phase 2)");
        param_layout = QGridLayout(param_group)
        param_layout.addWidget(QLabel("CNN Threshold (0-1):"), 0, 0)
        self.txt_cnn_threshold_display = QLineEdit(str(self.cnn_peak_threshold));
        self.txt_cnn_threshold_display.setReadOnly(True)
        param_layout.addWidget(self.txt_cnn_threshold_display, 0, 1)
        param_layout.addWidget(QLabel("Min Peak Dist (orig samples):"), 1, 0)  # Clarified this applies to original data
        self.txt_cnn_min_dist_display = QLineEdit(str(self.cnn_min_peak_distance));
        self.txt_cnn_min_dist_display.setReadOnly(True)
        param_layout.addWidget(self.txt_cnn_min_dist_display, 1, 1)
        left_layout.addWidget(param_group)

        left_layout.addStretch(1)
        log_group = QGroupBox("Log");
        log_v_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit();
        self.log_output.setReadOnly(True);
        self.log_output.setMinimumHeight(150)
        log_v_layout.addWidget(self.log_output)
        left_layout.addWidget(log_group)
        app_layout.addWidget(left_panel)

        plot_panel = QWidget();
        plot_v_layout = QVBoxLayout(plot_panel)
        plot_v_layout.addWidget(QLabel("View 1: Main View (Raw Data & CNN Output on Original Scale)"))
        self.main_plot_canvas = MplCanvas(self);
        plot_v_layout.addWidget(self.main_plot_canvas, stretch=3)
        plot_v_layout.addWidget(QLabel("View 2: CNN Processing View (Normalized Input & Direct Output)"))
        self.cnn_plot_canvas = MplCanvas(self);
        plot_v_layout.addWidget(self.cnn_plot_canvas, stretch=2)
        app_layout.addWidget(plot_panel, stretch=1)

        self.statusBar = QStatusBar();
        self.setStatusBar(self.statusBar)

    def log_message(self, message, status=False):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_output.append(timestamp + message)
        self.log_output.ensureCursorVisible()
        if status: self.statusBar.showMessage(message, 5000)
        utils.logging.info(message)

    def show_error_message_box(self, title, message):
        QMessageBox.critical(self, title, message)
        self.log_message(f"ERROR DIALOG: {title} - {message}")

    def load_csv_file_triggered(self):
        start_dir = os.path.dirname(self.csv_path) if self.csv_path and os.path.exists(
            os.path.dirname(self.csv_path)) else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Load FSEC CSV File", start_dir, "CSV Files (*.csv *.txt)")
        if path:
            self.csv_path = path
            try:
                temp_df = pd.read_csv(path, nrows=0, comment='#')
                if temp_df.columns.any():
                    first_col = temp_df.columns[0]
                    common_time_keywords = ["time", "elution", "volume", "min", "sec", "timepoint"]
                    if any(keyword in first_col.lower().replace("_", "").replace(" ", "") for keyword in
                           common_time_keywords):
                        self.timepoint_col_suggestion = first_col
                    else:
                        self.timepoint_col_suggestion = first_col
                    self.txt_timepoint_col.setText(self.timepoint_col_suggestion)
                else:
                    self.log_message(f"Could not read columns from CSV: {path}.")
            except Exception as e:
                self.log_message(f"Could not peek CSV header: {e}")
            self._process_loaded_csv_data()

    def update_data_after_timepoint_change(self):
        if self.csv_path:
            self.log_message(f"Timepoint column changed. Re-processing CSV.")
            self._process_loaded_csv_data()
        else:
            self.log_message("Timepoint column edited, but no CSV loaded.")

    def _process_loaded_csv_data(self):
        if not self.csv_path: self.show_error_message_box("Data Load Error", "No CSV file path."); return
        current_timepoint_col = self.txt_timepoint_col.text().strip()
        if not current_timepoint_col: self.show_error_message_box("Input Error", "Timepoint column name empty."); return

        self.log_message(f"Loading CSV: {self.csv_path} with timepoint: '{current_timepoint_col}'")
        try:
            loaded_data, identified_cols = utils.load_fsec_csv(self.csv_path, timepoint_col_name=current_timepoint_col)
        except Exception as e:
            self.show_error_message_box("CSV Load Failed", f"Error: {e}");
            utils.logging.error(f"CSV load error: {e}", exc_info=True)
            loaded_data, identified_cols = None, []

        if loaded_data:
            self.loaded_fsec_data = loaded_data;
            self.all_identified_sample_cols = identified_cols
            self.lbl_csv_path.setText(f"CSV: ...{os.path.basename(self.csv_path)}")
            self.log_message(f"Loaded {len(identified_cols)} samples.", status=True)
            self.populate_sample_dropdown()
            self.btn_view_all_inputs.setEnabled(True)
            if self.all_identified_sample_cols: QTimer.singleShot(100, self.plot_all_input_signals_triggered)
            self.check_enable_analysis_button()
        else:
            self.loaded_fsec_data = None;
            self.all_identified_sample_cols = []
            self.lbl_csv_path.setText("CSV: Load Error!");
            self.combo_samples.clear();
            self.current_selected_sample_name = None
            self.main_plot_canvas.clear_canvas("CSV Load Failed");
            self.cnn_plot_canvas.clear_canvas()
            self.btn_view_all_inputs.setEnabled(False);
            self.btn_analyze_sample.setEnabled(False)
            self.show_error_message_box("Data Load Info", f"Failed to load data from '{self.csv_path}'.")

    def populate_sample_dropdown(self):
        self.combo_samples.blockSignals(True);
        self.combo_samples.clear()
        if self.all_identified_sample_cols:
            self.combo_samples.addItems(self.all_identified_sample_cols)
            if self.all_identified_sample_cols: self.combo_samples.setCurrentIndex(0)
        self.combo_samples.blockSignals(False)
        if self.combo_samples.count() > 0:
            self.on_sample_selected_changed(self.combo_samples.currentText())
        else:
            self.main_plot_canvas.clear_canvas("No Samples Found");
            self.cnn_plot_canvas.clear_canvas()

    def on_sample_selected_changed(self, sample_name):
        self.current_sample_analysis_results = None
        if not self.loaded_fsec_data or not sample_name or sample_name not in self.loaded_fsec_data:
            self.current_selected_sample_name = None
            self.main_plot_canvas.clear_canvas(f"Sample '{sample_name}' not found.")
            self.cnn_plot_canvas.clear_canvas("Select a valid sample.")
            return

        self.current_selected_sample_name = sample_name
        x_data, y_data_raw = self.loaded_fsec_data[sample_name]
        y_data_numeric = np.array(y_data_raw, dtype=float);
        x_data_numeric = np.array(x_data, dtype=float)

        self.log_message(f"Displaying raw signal for: {sample_name}", status=True)
        try:
            fig_obj, _ = visualizations.plot_deconvolution_summary(
                original_x=x_data_numeric, original_y=y_data_numeric,
                title=f"Raw Signal: {sample_name}"
            )
            self.main_plot_canvas.update_plot(fig_obj)
            self.cnn_plot_canvas.clear_canvas("Run Analysis for CNN View")
        except Exception as e:
            self.log_message(f"Error plotting raw signal for {sample_name}: {e}", status=True);
            utils.logging.error("Raw plot error", exc_info=True)
            self.main_plot_canvas.clear_canvas(f"Plot Error for {sample_name}")
            self.cnn_plot_canvas.clear_canvas()

    def plot_all_input_signals_triggered(self):
        if not self.loaded_fsec_data or not self.all_identified_sample_cols:
            self.log_message("No data for overview.", status=True);
            self.main_plot_canvas.clear_canvas("No Data");
            return
        self.log_message(f"Plotting overview of all {len(self.loaded_fsec_data)} signals.", status=True)
        try:
            fig_obj, _ = visualizations.plot_raw_signals_overview(
                signals_dict=self.loaded_fsec_data,
                timepoint_col_name=self.txt_timepoint_col.text().strip(),
                title="All Input Signals Overview",
                highlight_sample=self.current_selected_sample_name
            )
            self.main_plot_canvas.update_plot(fig_obj)
            self.cnn_plot_canvas.clear_canvas("Overview Mode")
        except Exception as e:
            self.log_message(f"Error plotting all signals: {e}", status=True);
            utils.logging.error("All signals plot error", exc_info=True)
            self.main_plot_canvas.clear_canvas("Overview Plot Error");
            self.cnn_plot_canvas.clear_canvas()

    def load_model_folder_triggered(self):
        start_dir = os.path.dirname(self.model_folder_path) if self.model_folder_path and os.path.exists(
            os.path.dirname(self.model_folder_path)) else os.getcwd()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Trained Model Folder", start_dir)
        if folder_path:
            self.model_folder_path = folder_path
            self.log_message(f"Selected model folder: {folder_path}", status=True)
            model_weights_path, config_json_path = utils.get_model_and_config_paths(folder_path)

            if not config_json_path: self.show_error_message_box("Model Load Error",
                                                                 f"'config.json' not found."); return
            try:
                self.train_config = utils.load_config(config_json_path)
                self.log_message(f"Loaded training config: {config_json_path}")
            except Exception as e:
                self.show_error_message_box("Config Load Error", f"{e}"); self.train_config = None; return

            if not model_weights_path: self.show_error_message_box("Model Load Error",
                                                                   f"No '.pth' model found."); return
            if "model_params" not in self.train_config: self.show_error_message_box("Config Error",
                                                                                    "'model_params' missing."); self.train_config = None; return
            if "signal_length" not in self.train_config: self.show_error_message_box("Config Error",
                                                                                     "'signal_length' missing."); self.train_config = None; return
            if "generator_config" not in self.train_config or "x_range" not in self.train_config["generator_config"]:
                self.show_error_message_box("Config Error", "'generator_config' with 'x_range' missing.");
                self.train_config = None;
                return

            self.log_message(f"Attempting to load model weights: {model_weights_path}")
            self.cnn_model = load_model_for_gui(model_weights_path, self.train_config["model_params"], self.device)

            if self.cnn_model:
                model_name = os.path.basename(model_weights_path)
                self.lbl_model_path.setText(f"Model: {model_name}")
                self.log_message(f"CNN Model '{model_name}' loaded.", status=True)
                self.check_enable_analysis_button()
            else:
                self.lbl_model_path.setText("Model: Load Failed!");
                self.cnn_model = None
                self.show_error_message_box("Model Load Error", f"Failed to load '{model_weights_path}'.")

    def check_enable_analysis_button(self):
        if self.loaded_fsec_data and self.cnn_model and self.train_config:
            self.btn_analyze_sample.setEnabled(True)
            self.log_message("Data and model loaded. Ready for CNN analysis.", status=True)
        else:
            self.btn_analyze_sample.setEnabled(False)

    def run_analysis_for_selected_sample_triggered(self):
        if not self.current_selected_sample_name or not self.loaded_fsec_data: self.show_error_message_box(
            "Analysis Error", "No sample or data."); return
        if not self.cnn_model or not self.train_config: self.show_error_message_box("Analysis Error",
                                                                                    "Model or config not loaded."); return

        self.log_message(f"Starting CNN analysis for: {self.current_selected_sample_name}", status=True)
        x_original_unpadded, y_original_unpadded = self.loaded_fsec_data[self.current_selected_sample_name]

        try:
            model_params = self.train_config.get("model_params", {})
            generator_params = self.train_config.get("generator_config", {})

            model_depth = model_params.get("depth", 4)
            cnn_target_signal_length = self.train_config.get("signal_length")
            cnn_target_x_range = generator_params.get("x_range")

            if cnn_target_signal_length is None: self.show_error_message_box("Config Error",
                                                                             "'signal_length' missing."); return
            if cnn_target_x_range is None: self.show_error_message_box("Config Error",
                                                                       "'generator_config.x_range' missing."); return

            self.log_message(
                f"Preprocessing with CNN Target Length: {cnn_target_signal_length}, X-Range: {cnn_target_x_range}")

            prep_data = preprocess_for_cnn_and_fitter(
                x_original_unpadded, y_original_unpadded,
                model_depth, cnn_target_signal_length, cnn_target_x_range
            )
            if prep_data is None or len(prep_data["y_cnn_input_padded_norm"]) == 0:
                self.show_error_message_box("Preprocessing Error", "Preprocessing failed.");
                return

            input_tensor = torch.from_numpy(prep_data["y_cnn_input_padded_norm"]).float().unsqueeze(0).unsqueeze(0).to(
                self.device)
            with torch.no_grad():
                pred_tensor = self.cnn_model(input_tensor)

            output_activation_str = model_params.get("output_activation_str", "Sigmoid")
            pred_map_probs_padded = torch.sigmoid(pred_tensor) if output_activation_str == "None" else pred_tensor
            # Unpad to original_cnn_len_unpadded (which is cnn_target_signal_length)
            pred_map_probs_cnn_domain_unpadded = pred_map_probs_padded.squeeze().cpu().numpy()[
                                                 :prep_data["original_cnn_len_unpadded"]]

            # Interpolate probability map back to original X-axis
            prob_map_on_original_x = np.zeros_like(prep_data["x_original_unpadded"], dtype=float)
            if len(prep_data["x_cnn_canonical"]) > 1 and len(prep_data["x_original_unpadded"]) > 0 and \
                    len(pred_map_probs_cnn_domain_unpadded) == len(prep_data["x_cnn_canonical"]):

                # Ensure x_cnn_canonical is sorted (linspace should be, but good practice)
                # and its corresponding probabilities match
                sort_idx_can = np.argsort(prep_data["x_cnn_canonical"])
                x_can_sorted = prep_data["x_cnn_canonical"][sort_idx_can]
                probs_can_sorted = pred_map_probs_cnn_domain_unpadded[sort_idx_can]

                # Unique points for interpolation
                unique_x_can, unique_idx_can = np.unique(x_can_sorted, return_index=True)

                if len(unique_x_can) >= 2:  # Need at least 2 points for interp1d
                    interp_func_probs_to_orig = interp1d(
                        unique_x_can,
                        probs_can_sorted[unique_idx_can],
                        kind='linear',
                        fill_value="extrapolate",  # or (0.0, 0.0) to not extrapolate beyond learned range
                        bounds_error=False
                    )
                    # Query at original X points
                    prob_map_on_original_x = interp_func_probs_to_orig(prep_data["x_original_unpadded"])
                elif len(unique_x_can) == 1:  # Only one unique point in canonical domain
                    prob_map_on_original_x.fill(probs_can_sorted[unique_idx_can[0]])
                else:  # No points in canonical domain, fill with zeros
                    prob_map_on_original_x.fill(0.0)
            else:
                self.log_message(
                    "Warning: Mismatch in lengths for probability map interpolation. Resulting map may be inaccurate.",
                    status=True)
                if len(pred_map_probs_cnn_domain_unpadded) > 0:  # Fallback: fill with mean if possible
                    prob_map_on_original_x.fill(np.mean(pred_map_probs_cnn_domain_unpadded))

            prob_map_on_original_x = np.clip(prob_map_on_original_x, 0.0, 1.0)

            # Peak finding on the probability map that is now on the original X-axis scale
            detected_indices_on_original_x_cnn, _ = find_peaks(
                prob_map_on_original_x, height=self.cnn_peak_threshold, distance=self.cnn_min_peak_distance
            )
            cnn_detected_x_coords = prep_data["x_original_unpadded"][detected_indices_on_original_x_cnn] if len(
                detected_indices_on_original_x_cnn) > 0 else np.array([])

            self.log_message(f"CNN identified {len(cnn_detected_x_coords)} potential peaks.", status=True)

            self.current_sample_analysis_results = {
                "x_original": prep_data["x_original_unpadded"],
                "y_original": prep_data["y_original_unpadded"],
                "x_cnn_canonical": prep_data["x_cnn_canonical"],  # X-axis for CNN view
                "y_cnn_norm_unpadded": prep_data["y_cnn_input_padded_norm"][:prep_data["original_cnn_len_unpadded"]],
                # Y signal for CNN view
                "pred_map_probs_cnn_domain_unpadded": pred_map_probs_cnn_domain_unpadded,  # Prob map for CNN view
                "pred_map_probs_on_original_x": prob_map_on_original_x,  # Prob map for Main view
                "cnn_detected_peak_x_coords": cnn_detected_x_coords  # Peak X values for Main view
            }
            self.update_plots_after_analysis()

        except Exception as e:
            self.show_error_message_box("Analysis Failed", f"Error during CNN processing: {e}")
            utils.logging.error(f"CNN analysis error: {e}", exc_info=True)
            self.current_sample_analysis_results = None

    def update_plots_after_analysis(self):
        if not self.current_sample_analysis_results:
            self.log_message("No analysis results to plot.", status=True);
            return

        res = self.current_sample_analysis_results
        sample_name = self.current_selected_sample_name

        try:  # Main Plot
            fig_main, _ = visualizations.plot_deconvolution_summary(
                original_x=res["x_original"], original_y=res["y_original"],
                cnn_prob_map_original_x=res["pred_map_probs_on_original_x"],
                cnn_detected_peak_x_coords=res["cnn_detected_peak_x_coords"],
                title=f"Main View: {sample_name} (CNN Detections)"
            )
            self.main_plot_canvas.update_plot(fig_main)
        except Exception as e:
            self.log_message(f"Error updating main plot: {e}", status=True);
            utils.logging.error("Main plot error", exc_info=True)
            self.main_plot_canvas.clear_canvas(f"Main Plot Error")

        try:  # CNN Plot
            fig_cnn, _ = visualizations.plot_cnn_processing_view(
                cnn_input_x=res["x_cnn_canonical"],
                cnn_input_y_normalized=res["y_cnn_norm_unpadded"],
                cnn_output_prob_map=res["pred_map_probs_cnn_domain_unpadded"],
                cnn_threshold_value=self.cnn_peak_threshold,
                title=f"CNN View: {sample_name}"
            )
            self.cnn_plot_canvas.update_plot(fig_cnn)
        except Exception as e:
            self.log_message(f"Error updating CNN plot: {e}", status=True);
            utils.logging.error("CNN plot error", exc_info=True)
            self.cnn_plot_canvas.clear_canvas(f"CNN Plot Error")
        self.log_message("Analysis plots updated.", status=True)

    def closeEvent(self, event):
        self.log_message("Application closing.", status=True);
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Basic logging config if not already set by utils or other means
    if not utils.logging.getLogger().hasHandlers() and \
            not (hasattr(utils, 'logging') and utils.logging.getLogger(utils.__name__).hasHandlers()):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        utils.logging = logging  # Make it accessible via utils.logging if it was just basicConfig'd

    main_window = FSECAnalysisGUI()
    main_window.show()
    sys.exit(app.exec_())
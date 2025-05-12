# dummy_gui.py
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.special import erf  # For dummy EMG data generation

# --- Assume visualizations.py is in the same directory or accessible in PYTHONPATH ---
try:
    import visualizations
except ImportError:
    try:
        from fseccnn import visualizations
    except ImportError:
        print(
            "CRITICAL: visualizations.py not found. Make sure it's in the same directory or fseccnn is in PYTHONPATH.")
        sys.exit(1)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Figure is created here and passed to superclass constructor
        self.figure = plt.Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # Add an initial blank plot to the figure's axes
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Canvas Initialized", ha='center', va='center', color='grey')
        self.draw()


class DummyFSECAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dummy Visualization Tester")
        self.setGeometry(100, 100, 1200, 800)

        self.initUI()
        print("Dummy GUI Initialized. Click buttons to test plots.")

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(250)

        btn_test_raw_overview = QPushButton("Test Raw Overview Plot")
        btn_test_raw_overview.clicked.connect(self.test_plot_raw_overview)
        control_layout.addWidget(btn_test_raw_overview)

        btn_test_cnn_view = QPushButton("Test CNN View Plot")
        btn_test_cnn_view.clicked.connect(self.test_plot_cnn_view)
        control_layout.addWidget(btn_test_cnn_view)

        btn_test_deconv_summary = QPushButton("Test Deconvolution Plot")
        btn_test_deconv_summary.clicked.connect(self.test_plot_deconvolution_summary)
        control_layout.addWidget(btn_test_deconv_summary)

        btn_clear_plots = QPushButton("Clear Plots")
        btn_clear_plots.clicked.connect(self.clear_all_plots)
        control_layout.addWidget(btn_clear_plots)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        plot_layout.addWidget(QLabel("Plot 1 (e.g., Deconv/Raw Overview):"))
        self.plot_canvas1 = MplCanvas(self, width=7, height=5, dpi=100)
        plot_layout.addWidget(self.plot_canvas1)

        plot_layout.addWidget(QLabel("Plot 2 (e.g., CNN View):"))
        self.plot_canvas2 = MplCanvas(self, width=7, height=4, dpi=100)
        plot_layout.addWidget(self.plot_canvas2)

        main_layout.addWidget(plot_panel, stretch=1)

    def _generate_dummy_chromatogram_data(self, num_points=500, num_peaks=2, add_noise=True, y_offset=0, x_scale=100):
        x = np.linspace(0, x_scale, num_points)
        y = np.zeros_like(x) + y_offset
        peak_centers = np.linspace(x_scale * 0.2, x_scale * 0.8, num_peaks)
        amplitude_base = 10
        for i, center in enumerate(peak_centers):
            amplitude = amplitude_base * (1 - 0.3 * i)  # Vary amplitude slightly
            y += amplitude * np.exp(-0.01 * (x_scale / 100) * (x - center) ** 2)  # Simple Gaussian
        if add_noise:
            y += np.random.normal(0, 0.5, num_points)
        return x, y

    def test_plot_raw_overview(self):
        print("Testing Raw Overview Plot...")
        signals = {}
        for i in range(3):
            x, y = self._generate_dummy_chromatogram_data(num_peaks=(i % 3) + 1, y_offset=i * 2)
            signals[f"Sample_{chr(65 + i)}"] = (x, y)

        try:
            new_fig, ax = visualizations.plot_raw_signals_overview(
                signals,
                timepoint_col_name="Dummy Time",
                title="Dummy Raw Signals Overview",
                highlight_sample="Sample_B"
            )
            self.plot_canvas1.figure = new_fig  # Assign the new figure
            self.plot_canvas1.draw()
            print("Raw Overview Plot updated to plot_canvas1.")
        except Exception as e:
            print(f"Error in test_plot_raw_overview: {e}")
            self._show_error_on_canvas(self.plot_canvas1, str(e))

    def test_plot_cnn_view(self):
        print("Testing CNN View Plot...")
        x_cnn = np.linspace(0, 1, 256)  # Canonical X for CNN
        y_cnn_norm = np.sin(x_cnn * 2 * np.pi) * 0.4 + 0.5
        y_cnn_norm = np.clip(y_cnn_norm, 0, 1)
        prob_map = np.zeros_like(x_cnn)
        prob_map[50:70] = np.linspace(0, 0.8, 20) ** 2
        prob_map[70:90] = np.linspace(0.8, 0, 20) ** 2
        prob_map[150:180] = 0.6 * np.exp(-0.1 * (np.arange(30) - 15) ** 2)

        try:
            new_fig, ax = visualizations.plot_cnn_processing_view(
                cnn_input_x=x_cnn,
                cnn_input_y_normalized=y_cnn_norm,
                cnn_output_prob_map=prob_map,
                cnn_threshold_value=0.3,
                title="Dummy CNN Processing View"
            )
            self.plot_canvas2.figure = new_fig
            self.plot_canvas2.draw()
            print("CNN View Plot updated to plot_canvas2.")
        except Exception as e:
            print(f"Error in test_plot_cnn_view: {e}")
            self._show_error_on_canvas(self.plot_canvas2, str(e))

    def test_plot_deconvolution_summary(self):
        print("Testing Deconvolution Summary Plot...")
        original_x, original_y = self._generate_dummy_chromatogram_data(num_points=500, num_peaks=2, x_scale=100)

        fitted_baseline_y = np.ones_like(original_x) * np.min(original_y) * 0.1 + original_x * 0.005

        comp1_center = 30
        comp1_amp = 8
        comp1_y = comp1_amp * np.exp(-0.01 * (original_x - comp1_center) ** 2)

        comp2_center = 65
        comp2_amp = 10
        comp2_sigma = 5
        comp2_tau = 3
        _t_comp2 = (original_x - comp2_center) / comp2_tau
        erf_input = ((comp2_sigma / (np.sqrt(2) * comp2_tau)) - (_t_comp2 / np.sqrt(2)))
        comp2_y = (comp2_amp / (2 * comp2_tau)) * np.exp((_t_comp2 / 2) + (comp2_sigma ** 2 / (2 * comp2_tau ** 2))) * \
                  (1 - erf(erf_input))

        individual_components = [
            {'x_coords': original_x, 'y_coords_on_baseline': comp1_y + fitted_baseline_y,
             'label': f'Fit 1 (μ:{comp1_center:.1f}, A:{comp1_amp:.1f})'},
            {'x_coords': original_x, 'y_coords_on_baseline': comp2_y + fitted_baseline_y,
             'label': f'Fit 2 (μ:{comp2_center:.1f}, τ:{comp2_tau:.1f})'}
        ]
        fitted_total_y = comp1_y + comp2_y + fitted_baseline_y

        cnn_prob_map = np.zeros_like(original_x)
        cnn_prob_map[int(len(original_x) * 0.2): int(len(original_x) * 0.2) + 50] = np.linspace(0, 0.7, 50) ** 2
        cnn_prob_map[int(len(original_x) * 0.2) + 50: int(len(original_x) * 0.2) + 100] = np.linspace(0.7, 0, 50) ** 2
        cnn_prob_map[int(len(original_x) * 0.6): int(len(original_x) * 0.6) + 60] = 0.5 * np.exp(
            -0.05 * (np.arange(60) - 30) ** 2)

        # This is the variable that should be used
        cnn_detections_x = original_x[[int(len(original_x) * 0.25), int(len(original_x) * 0.65)]]

        try:
            new_fig, ax = visualizations.plot_deconvolution_summary(
                original_x=original_x,
                original_y=original_y,
                fitted_total_y=fitted_total_y,
                individual_emg_components_data=individual_components,
                fitted_baseline_y=fitted_baseline_y,
                cnn_prob_map_original_x=cnn_prob_map,
                cnn_detected_peak_x_coords=cnn_detections_x,  # CORRECTED HERE
                title="Dummy Deconvolution Summary"
            )
            self.plot_canvas1.figure = new_fig
            self.plot_canvas1.draw()
            print("Deconvolution Summary Plot updated to plot_canvas1.")
        except Exception as e:
            print(f"Error in test_plot_deconvolution_summary: {e}")
            # Show error on canvas if plotting fails
            if hasattr(self, 'plot_canvas1'):  # ensure canvas exists
                self._show_error_on_canvas(self.plot_canvas1, f"Plotting Error: {e}")
            else:
                print(f"Plot canvas 1 not available to show error: {e}")

    def clear_all_plots(self):
        print("Clearing plots...")
        for canvas in [self.plot_canvas1, self.plot_canvas2]:
            if hasattr(canvas, 'figure') and canvas.figure is not None:
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, "Plot Cleared", ha='center', va='center', color='grey')
                canvas.draw()
        print("Plots cleared.")

    def _show_error_on_canvas(self, canvas, error_message):
        if hasattr(canvas, 'figure') and canvas.figure is not None:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot Error:\n{error_message}",
                    ha='center', va='center', color='red', wrap=True, fontsize=9)
            canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DummyFSECAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())
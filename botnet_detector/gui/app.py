import queue
from tkinter import messagebox

from botnet_detector.config import APP_TITLE
from botnet_detector.dataset import DatasetMixin
from botnet_detector.model_io import ModelIOMixin
from botnet_detector.pcap import PcapMixin
from botnet_detector.plots import PlotsMixin
from botnet_detector.prediction import PredictionMixin
from botnet_detector.training import TrainingMixin
from botnet_detector.gui.dashboard import DashboardMixin
from botnet_detector.gui.predict_tab import PredictTabMixin
from botnet_detector.gui.styles import StyleMixin
from botnet_detector.gui.train_tab import TrainTabMixin


class BotnetDetectionApp(
    StyleMixin,
    DashboardMixin,
    TrainTabMixin,
    PredictTabMixin,
    DatasetMixin,
    PcapMixin,
    TrainingMixin,
    ModelIOMixin,
    PredictionMixin,
    PlotsMixin,
):
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1400x900")
        self.root.minsize(1150, 720)

        self.df = None
        self.processed_df = None
        self.scaler = None
        self.selector = None
        self.ensemble = None
        self.selected_features = []
        self.feature_columns = []
        self.train_medians = {}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.metrics = {}
        self.model_metrics = {}
        self.feature_importances = {}
        self.label_counts = {}
        self.raw_label_counts = {}
        self.roc_data = None
        self.pr_data = None
        self.predictions_df = None

        self._busy = False
        self.msg_queue = queue.Queue()

        self.configure_styles()
        self.build_header()
        self.build_notebook()
        self.build_dashboard()
        self.build_train_tab()
        self.build_predict_tab()

        self.status_var.set("Ready — Load a labelled traffic CSV dataset to begin.")
        self.root.after(150, self._poll_queue)

    def log(self, text):
        self.log_box.insert("end", str(text) + "\n")
        self.log_box.see("end")
        self.root.update_idletasks()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "train_done":
                    self._apply_training_result(payload)
                elif kind == "train_error":
                    self._busy = False
                    self.progress.stop()
                    self.load_btn.config(state="normal")
                    self.train_btn.config(state="normal")
                    self.status_var.set("Training failed.")
                    messagebox.showerror("Training Error", payload)
                elif kind == "lc_done":
                    self._busy = False
                    self.progress.stop()
                    self.status_var.set("Learning curve complete.")
                    self._draw_learning_curve(*payload)
                elif kind == "lc_error":
                    self._busy = False
                    self.progress.stop()
                    self.status_var.set("Learning curve failed.")
                    messagebox.showerror("Learning Curve Error", payload)
                elif kind == "pcap_done":
                    self._busy = False
                    self.progress.stop()
                    self.load_btn.config(state="normal")
                    self.load_pcap_btn.config(state="normal")
                    self.df = payload
                    self.log(f"Extracted {len(self.df)} flows from PCAP")
                    counts = self.df["Label"].value_counts().to_dict()
                    self.log(f"Label counts: {counts}")
                    self.train_btn.config(state="normal")
                    self.update_dashboard()
                    self.status_var.set(f"PCAP loaded — {len(self.df)} flows extracted")
                elif kind == "pcap_error":
                    self._busy = False
                    self.progress.stop()
                    self.load_btn.config(state="normal")
                    self.load_pcap_btn.config(state="normal")
                    self.status_var.set("PCAP loading failed.")
                    messagebox.showerror("PCAP Loading Error", payload)
                elif kind == "pcap_predict_done":
                    self._busy = False
                    self.progress.stop()
                    self.status_var.set(f"PCAP prediction ready — {len(payload)} flows")
                    self._run_prediction_on_dataframe(payload)
                elif kind == "pcap_predict_error":
                    self._busy = False
                    self.progress.stop()
                    self.status_var.set("PCAP prediction failed.")
                    messagebox.showerror("PCAP Prediction Error", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

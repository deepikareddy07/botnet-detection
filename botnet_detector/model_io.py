import os
from tkinter import filedialog, messagebox

import joblib


class ModelIOMixin:
    def save_model(self):
        if self.ensemble is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save Model Bundle",
            defaultextension=".pkl",
            filetypes=[
                ("Pickle files", "*.pkl")
            ]
        )

        if not path:
            return

        bundle = {
            "ensemble": self.ensemble,
            "scaler": self.scaler,
            "selector": self.selector,
            "selected_features": self.selected_features,
            "feature_columns": self.feature_columns,
            "train_medians": self.train_medians
        }

        try:
            joblib.dump(
                bundle,
                path
            )

            messagebox.showinfo(
                "Saved",
                f"Model saved successfully:\n{path}"
            )

        except Exception as e:
            messagebox.showerror(
                "Save Error",
                str(e)
            )

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Load Model Bundle",
            filetypes=[
                ("Pickle files", "*.pkl")
            ]
        )

        if not path:
            return

        try:
            bundle = joblib.load(path)

            self.ensemble = bundle["ensemble"]
            self.scaler = bundle["scaler"]
            self.selector = bundle["selector"]
            self.selected_features = (
                bundle["selected_features"]
            )
            self.feature_columns = (
                bundle["feature_columns"]
            )
            self.train_medians = bundle.get("train_medians", {})

            self.save_pred_btn.config(state="disabled")

            messagebox.showinfo(
                "Model Loaded",
                "Model bundle loaded successfully. Note: charts that need "
                "evaluation results (metrics, ROC, dataset insights) will "
                "only populate once you train in this session — a loaded "
                "model can be used for prediction right away though."
            )

            self.status_var.set(
                f"Model loaded — {os.path.basename(path)}"
            )

        except Exception as e:
            messagebox.showerror(
                "Load Error",
                str(e)
            )

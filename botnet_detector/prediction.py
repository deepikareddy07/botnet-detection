from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd


class PredictionMixin:
    def predict_csv(self):
        if self.ensemble is None:
            messagebox.showwarning(
                "No Model",
                "Train or load a model first."
            )
            return

        path = filedialog.askopenfilename(
            title="Select Traffic CSV",
            filetypes=[
                ("CSV files", "*.csv")
            ]
        )

        if not path:
            return

        try:
            new_df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        self._run_prediction_on_dataframe(new_df)

    def _run_prediction_on_dataframe(self, new_df):
        """Shared by both predict_csv() and the PCAP prediction path —
        given a dataframe of flow features, scale/select/predict and
        populate the results table exactly the same way either way."""
        try:
            work = new_df.copy()

            work.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True
            )
            work.fillna(0, inplace=True)

            if "Label" in work.columns:
                work = work.drop(
                    columns=["Label"]
                )

            missing = [
                c for c in self.feature_columns
                if c not in work.columns
            ]

            if missing:
                messagebox.showerror(
                    "Column Mismatch",
                    "Missing required columns:\n"
                    + "\n".join(
                        missing[:20]
                    )
                )
                return

            work = work[
                self.feature_columns
            ]

            for c in work.columns:
                work[c] = pd.to_numeric(
                    work[c],
                    errors="coerce"
                )

            median_series = pd.Series(self.train_medians)
            work = work.fillna(median_series).fillna(0)

            X_s = self.scaler.transform(
                work
            )

            X_sel = self.selector.transform(
                X_s
            )

            preds = self.ensemble.predict(
                X_sel
            )

        except Exception as e:
            messagebox.showerror(
                "Prediction Error",
                str(e)
            )
            return

        labels = np.where(
            preds == 1,
            "BOTNET",
            "BENIGN"
        )

        self.predictions_df = new_df.copy()
        self.predictions_df[
            "Prediction"
        ] = labels

        botnet_count = int(
            (preds == 1).sum()
        )
        benign_count = int(
            (preds == 0).sum()
        )

        self.pred_summary.config(
            text=(
                f"TOTAL FLOWS: {len(preds):,}    "
                f"• BENIGN: {benign_count:,}    "
                f"• BOTNET: {botnet_count:,}"
            ),
            fg="#1565c0"
        )

        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)

        for i, label in enumerate(
            labels[:1000]
        ):
            tag = (
                "attack"
                if label == "BOTNET"
                else "normal"
            )

            self.pred_tree.insert(
                "",
                "end",
                values=(i, label),
                tags=(tag,)
            )

        self.save_pred_btn.config(
            state="normal"
        )
        self.pred_chart_btn.config(
            state="normal"
        )
        self._last_prediction_counts = (benign_count, botnet_count)

        self.card_values["attacks"].config(
            text=f"{botnet_count:,}"
        )
        self.card_values["normal"].config(
            text=f"{benign_count:,}"
        )

        self.dashboard_status.config(
            text=(
                "PREDICTION COMPLETE — "
                "BENIGN/BOTNET results generated"
            ),
            bg="#e8f5e9",
            fg="#1b5e20"
        )

        self.status_var.set(
            f"Prediction completed — {len(preds):,} flows analyzed."
        )

    def save_predictions(self):
        if self.predictions_df is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save Predictions",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv")
            ]
        )

        if not path:
            return

        try:
            self.predictions_df.to_csv(
                path,
                index=False
            )

            messagebox.showinfo(
                "Saved",
                f"Predictions saved to:\n{path}"
            )

        except Exception as e:
            messagebox.showerror(
                "Save Error",
                str(e)
            )

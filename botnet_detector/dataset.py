import os
from tkinter import filedialog, messagebox

import pandas as pd


class DatasetMixin:
    def load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select traffic CSV dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            self.df = pd.read_csv(path, low_memory=False)
            self.df.columns = [str(c).strip() for c in self.df.columns]
        except Exception as e:
            messagebox.showerror(
                "Load Error",
                str(e)
            )
            return

        self.log_box.delete("1.0", "end")
        self.log(
            f"Loaded dataset: {os.path.basename(path)}"
        )
        self.log(
            f"Shape: {self.df.shape}"
        )
        self.log(
            f"Columns: {list(self.df.columns)}"
        )

        label_candidates = {str(c).strip().lower(): c for c in self.df.columns}
        if "label" not in label_candidates:
            messagebox.showwarning(
                "Label Missing",
                "Training requires a labelled dataset with a Label column."
            )
            self.train_btn.config(state="disabled")
        else:
            actual_label = label_candidates["label"]
            if actual_label != "Label":
                self.df.rename(columns={actual_label: "Label"}, inplace=True)
            self.train_btn.config(state="normal")

        self.update_dashboard()

        self.status_var.set(
            f"Dataset loaded — {os.path.basename(path)}"
        )

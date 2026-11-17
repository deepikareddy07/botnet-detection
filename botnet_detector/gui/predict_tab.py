import tkinter as tk
from tkinter import ttk


class PredictTabMixin:
    def build_predict_tab(self):
        frame = self.predict_tab
        PAGE_BG = "#eef1f8"

        page = tk.Frame(frame, bg=PAGE_BG)
        page.pack(fill="both", expand=True)

        hero = tk.Frame(page, bg="#1c2b6b")
        hero.pack(fill="x", padx=20, pady=(18, 6))
        hero_inner = tk.Frame(hero, bg="#1c2b6b")
        hero_inner.pack(fill="x", padx=24, pady=16)
        tk.Label(
            hero_inner, text="Predict on New Traffic",
            bg="#1c2b6b", fg="white", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")
        tk.Label(
            hero_inner,
            text="Classify unlabelled flows with the trained ensemble — CSV or raw PCAP.",
            bg="#1c2b6b", fg="#c7d1f5", font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(3, 0))

        self._section_header(page, PAGE_BG, "PREDICTION CONTROLS & RESULTS", "#C2185B")
        body = tk.Frame(page, bg=PAGE_BG)
        body.pack(fill="both", expand=True, padx=0, pady=(0, 18))

        controls_panel = tk.Frame(body, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        controls_panel.pack(side="left", fill="y", padx=(0, 12))
        tk.Frame(controls_panel, bg="#C2185B", height=4).pack(fill="x")
        tk.Label(
            controls_panel, text="Prediction Controls", bg="white", fg="#1c2b45",
            font=("Segoe UI", 10, "bold"), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        ttk.Button(
            controls_panel, text="Load Trained Model", command=self.load_model
        ).pack(fill="x", padx=10, pady=4)

        ttk.Button(
            controls_panel, text="Load Traffic CSV & Predict", command=self.predict_csv
        ).pack(fill="x", padx=10, pady=4)

        ttk.Button(
            controls_panel, text="Load PCAP & Predict", command=self.load_pcap_predict
        ).pack(fill="x", padx=10, pady=4)

        self.save_pred_btn = ttk.Button(
            controls_panel, text="Save Predictions",
            command=self.save_predictions, state="disabled"
        )
        self.save_pred_btn.pack(fill="x", padx=10, pady=4)

        self.pred_chart_btn = ttk.Button(
            controls_panel, text="Results Pie Chart",
            command=self.show_prediction_pie, state="disabled"
        )
        self.pred_chart_btn.pack(fill="x", padx=10, pady=4)
        tk.Frame(controls_panel, bg="white", height=10).pack()

        content = tk.Frame(body, bg=PAGE_BG)
        content.pack(side="left", fill="both", expand=True)

        summary_panel = tk.Frame(content, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        summary_panel.pack(fill="x", pady=(0, 12))
        tk.Frame(summary_panel, bg="#2874A6", height=4).pack(fill="x")
        self.pred_summary = tk.Label(
            summary_panel, text="No predictions yet.",
            bg="white", fg="#1565c0", font=("Segoe UI", 15, "bold"),
            padx=12, pady=13
        )
        self.pred_summary.pack(fill="x")

        results_panel = tk.Frame(content, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        results_panel.pack(fill="both", expand=True)
        tk.Frame(results_panel, bg="#27703B", height=4).pack(fill="x")
        tk.Label(
            results_panel, text="Prediction Results", bg="white", fg="#1c2b45",
            font=("Segoe UI", 10, "bold"), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        table_frame = tk.Frame(results_panel, bg="white")
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
        scroll_y.pack(side="right", fill="y")

        self.pred_tree = ttk.Treeview(
            table_frame, columns=("row", "prediction"),
            show="headings", yscrollcommand=scroll_y.set
        )
        scroll_y.config(command=self.pred_tree.yview)

        self.pred_tree.heading("row", text="Row #")
        self.pred_tree.heading("prediction", text="Prediction")
        self.pred_tree.column("row", width=100, anchor="center")
        self.pred_tree.column("prediction", width=300, anchor="center")

        self.pred_tree.tag_configure("attack", background="#ffcdd2", foreground="#b71c1c")
        self.pred_tree.tag_configure("normal", background="#c8e6c9", foreground="#1b5e20")

        self.pred_tree.pack(fill="both", expand=True)

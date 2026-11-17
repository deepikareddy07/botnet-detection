import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class TrainTabMixin:
    def build_train_tab(self):
        frame = self.train_tab
        PAGE_BG = "#eef1f8"

        page = tk.Frame(frame, bg=PAGE_BG)
        page.pack(fill="both", expand=True)

        hero = tk.Frame(page, bg="#1c2b6b")
        hero.pack(fill="x", padx=20, pady=(18, 6))
        hero_inner = tk.Frame(hero, bg="#1c2b6b")
        hero_inner.pack(fill="x", padx=24, pady=16)
        tk.Label(
            hero_inner, text="Model Training & Evaluation",
            bg="#1c2b6b", fg="white", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")
        tk.Label(
            hero_inner,
            text="Load traffic data, train the weighted voting ensemble, and inspect results.",
            bg="#1c2b6b", fg="#c7d1f5", font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(3, 0))

        self._section_header(page, PAGE_BG, "DATASET & MODEL CONTROLS", "#2874A6")
        controls_panel = tk.Frame(page, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        controls_panel.pack(fill="x", padx=0, pady=(0, 4))
        tk.Frame(controls_panel, bg="#2874A6", height=4).pack(fill="x")
        controls = tk.Frame(controls_panel, bg="white")
        controls.pack(fill="x", padx=14, pady=12)

        self.load_btn = ttk.Button(controls, text="Load Dataset CSV", command=self.load_dataset)
        self.load_btn.grid(row=0, column=0, padx=8, pady=4)

        self.load_pcap_btn = ttk.Button(controls, text="Load PCAP File", command=self.load_pcap_dataset)
        self.load_pcap_btn.grid(row=0, column=1, padx=8, pady=4)

        self.train_btn = ttk.Button(
            controls, text="Preprocess + Train Ensemble",
            command=self.start_training, state="disabled"
        )
        self.train_btn.grid(row=0, column=2, padx=8, pady=4)

        self.save_model_btn = ttk.Button(
            controls, text="Save Model", command=self.save_model, state="disabled"
        )
        self.save_model_btn.grid(row=0, column=3, padx=8, pady=4)

        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.grid(row=0, column=4, sticky="ew", padx=10)
        controls.grid_columnconfigure(4, weight=1)

        self._section_header(page, PAGE_BG, "VISUALIZATIONS & TRAINING LOG", "#7B3FA0")
        viz_body = tk.Frame(page, bg=PAGE_BG)
        viz_body.pack(fill="both", expand=True, padx=0, pady=(0, 18))

        viz_panel = tk.Frame(viz_body, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        viz_panel.pack(side="left", fill="y", padx=(0, 12))
        tk.Frame(viz_panel, bg="#7B3FA0", height=4).pack(fill="x")
        tk.Label(
            viz_panel, text="Model Visualizations", bg="white", fg="#1c2b45",
            font=("Segoe UI", 10, "bold"), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        graph_specs = [
            ("Class Distribution", self.show_class_distribution),
            ("Metrics Comparison", self.show_metrics_chart),
            ("Feature Importance", self.show_feature_importance),
            ("Label Distribution", self.show_attack_type_breakdown),
            ("PCA Scatter", self.show_pca_scatter),
            ("ROC / AUC", self.show_roc_curve),
            ("Precision-Recall", self.show_pr_curve),
            ("Learning Curve", self.show_learning_curve),
            ("Flow Duration", self.show_duration_distribution),
            ("Correlation Heatmap", self.show_correlation_heatmap)
        ]
        self.graph_buttons = []
        for text, command in graph_specs:
            btn = ttk.Button(viz_panel, text=text, command=command, state="disabled")
            btn.pack(fill="x", padx=10, pady=4)
            self.graph_buttons.append(btn)
        tk.Frame(viz_panel, bg="white", height=10).pack()

        log_panel = tk.Frame(viz_body, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        log_panel.pack(side="left", fill="both", expand=True)
        tk.Frame(log_panel, bg="#27703B", height=4).pack(fill="x")
        tk.Label(
            log_panel, text="Training Log & Evaluation", bg="white", fg="#1c2b45",
            font=("Segoe UI", 10, "bold"), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        self.log_box = ScrolledText(
            log_panel, bg="#172a3a", fg="#dff6ff",
            insertbackground="white", font=("Consolas", 10), relief="flat"
        )
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

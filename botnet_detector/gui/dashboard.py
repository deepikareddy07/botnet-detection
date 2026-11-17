import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class DashboardMixin:
    @staticmethod
    def _round_rect_points(x0, y0, x1, y1, r):
        return [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]

    def _rounded_card(self, parent, width, height, bg_page, card_fill, accent, r=14):
        """Draws a soft-shadowed, rounded-corner card on a Canvas (Tkinter
        has no native rounded-rect/shadow support) and returns (canvas,
        content_frame) so callers can pack normal widgets — icon, label,
        value — inside content_frame exactly like a regular Frame."""
        canvas = tk.Canvas(
            parent, width=width, height=height,
            bg=bg_page, highlightthickness=0, bd=0
        )
        shadow_pts = self._round_rect_points(6, 8, width - 2, height + 2, r)
        canvas.create_polygon(shadow_pts, smooth=True, fill="#d7deea", outline="")
        card_pts = self._round_rect_points(3, 3, width - 5, height - 3, r)
        canvas.create_polygon(card_pts, smooth=True, fill=card_fill, outline="")
        canvas.create_rectangle(3, 3, 8, height - 3, fill=accent, outline="")

        content = tk.Frame(canvas, bg=card_fill)
        canvas.create_window(20, height / 2, window=content, anchor="w", width=width - 34)
        return canvas, content

    def _section_header(self, parent, bg_page, text, accent):
        row = tk.Frame(parent, bg=bg_page)
        row.pack(fill="x", pady=(14, 6))
        tk.Frame(row, bg=accent, width=4, height=18).pack(side="left", padx=(0, 8))
        tk.Label(
            row, text=text, bg=bg_page, fg="#1c2b45",
            font=("Segoe UI", 12, "bold")
        ).pack(side="left")

    def build_dashboard(self):
        frame = self.dashboard_tab
        PAGE_BG = "#eef1f8"
        frame.configure(style="Dash.TFrame")
        style = ttk.Style()
        style.configure("Dash.TFrame", background=PAGE_BG)

        outer = tk.Frame(frame, bg=PAGE_BG)
        outer.pack(fill="both", expand=True)

        canvas_scroll = tk.Canvas(outer, bg=PAGE_BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        page = tk.Frame(canvas_scroll, bg=PAGE_BG)
        page_window = canvas_scroll.create_window((0, 0), window=page, anchor="nw")

        def _on_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
            canvas_scroll.itemconfig(page_window, width=event.width)
        canvas_scroll.bind("<Configure>", _on_configure)
        page.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))

        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        # ---- Hero banner ----
        hero = tk.Frame(page, bg="#1c2b6b")
        hero.pack(fill="x", padx=20, pady=(18, 6))
        hero_inner = tk.Frame(hero, bg="#1c2b6b")
        hero_inner.pack(fill="x", padx=24, pady=18)

        tk.Label(
            hero_inner, text="Security Analytics Dashboard",
            bg="#1c2b6b", fg="white", font=("Segoe UI", 21, "bold")
        ).pack(anchor="w")
        tk.Label(
            hero_inner,
            text="Live overview of dataset composition, model performance, and detection status.",
            bg="#1c2b6b", fg="#c7d1f5", font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(4, 0))

        self.dashboard_status = tk.Label(
            hero_inner, text="WAITING — Load a labelled dataset",
            bg="#2c3d8f", fg="#e8ecff", font=("Segoe UI", 10, "bold"),
            padx=14, pady=6
        )
        self.dashboard_status.pack(anchor="w", pady=(12, 0))

        self.card_values = {}

        def make_row(parent, section_title, section_accent, specs):
            self._section_header(parent, PAGE_BG, section_title, section_accent)
            row = tk.Frame(parent, bg=PAGE_BG)
            row.pack(fill="x")
            for i, (label, key, accent, tint) in enumerate(specs):
                canvas, content = self._rounded_card(row, 260, 86, PAGE_BG, tint, accent)
                canvas.grid(row=0, column=i, padx=6, pady=2, sticky="nsew")
                row.grid_columnconfigure(i, weight=1)

                tk.Label(
                    content, text=label, bg=tint, fg="#5b6b8c",
                    font=("Segoe UI", 9, "bold")
                ).pack(anchor="w")
                value = tk.Label(
                    content, text="—", bg=tint, fg=accent,
                    font=("Segoe UI", 21, "bold")
                )
                value.pack(anchor="w", pady=(2, 0))
                self.card_values[key] = value

        make_row(page, "DATASET OVERVIEW", "#2874A6", [
            ("Dataset Flows", "dataset", "#2874A6", "#E4EEFB"),
            ("Features", "features", "#546E7A", "#ECEFF1"),
            ("BENIGN", "normal", "#2874A6", "#E4EEFB"),
            ("BOTNET", "attacks", "#C0392B", "#FBD7D7"),
        ])

        make_row(page, "MODEL PERFORMANCE", "#27703B", [
            ("Accuracy", "accuracy", "#27703B", "#E4F2E0"),
            ("Precision", "precision", "#148F77", "#D9F2ED"),
            ("Recall", "recall", "#C99A2E", "#FDF1D6"),
            ("F1 Score", "f1", "#7B3FA0", "#F1E6F8"),
        ])

        make_row(page, "CONFUSION MATRIX", "#C2185B", [
            ("True Positives", "tp", "#27703B", "#E4F2E0"),
            ("True Negatives", "tn", "#2874A6", "#E4EEFB"),
            ("False Positives", "fp", "#C99A2E", "#FDF1D6"),
            ("False Negatives", "fn", "#C0392B", "#FBD7D7"),
        ])

        # ---- Quick actions + activity log ----
        self._section_header(page, PAGE_BG, "QUICK ACTIONS & ACTIVITY", "#7B3FA0")
        body = tk.Frame(page, bg=PAGE_BG)
        body.pack(fill="both", expand=True, pady=(0, 20))

        actions_panel = tk.Frame(body, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        actions_panel.pack(side="left", fill="y", padx=(0, 12))
        tk.Frame(actions_panel, bg="#7B3FA0", height=4).pack(fill="x")
        tk.Label(
            actions_panel, text="Quick Visual Analytics", bg="white", fg="#1c2b45",
            font=("Segoe UI", 10, "bold"), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        viz_buttons = [
            ("Class Split", self.show_class_pie),
            ("Confusion Matrix", self.show_confusion_matrix),
            ("Model Comparison", self.show_model_comparison)
        ]
        self.dashboard_buttons = []
        for text, command in viz_buttons:
            btn = ttk.Button(actions_panel, text=text, command=command, state="disabled")
            btn.pack(fill="x", padx=10, pady=5)
            self.dashboard_buttons.append(btn)
        tk.Frame(actions_panel, bg="white", height=10).pack()

        activity_panel = tk.Frame(body, bg="white", highlightbackground="#d5e0e8", highlightthickness=1)
        activity_panel.pack(side="left", fill="both", expand=True)
        tk.Frame(activity_panel, bg="#2874A6", height=4).pack(fill="x")
        tk.Label(
            activity_panel, text="Security Activity", bg="white", fg="#1c2b45",
            font=("Segoe UI", 10, "bold"), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        self.activity_text = ScrolledText(
            activity_panel, height=7, bg="#102a43", fg="#d9eaf7",
            insertbackground="white", font=("Consolas", 10), relief="flat"
        )
        self.activity_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.activity_text.insert(
            "end",
            "Dashboard initialized.\n"
            "Load a labelled traffic dataset to begin.\n"
        )
        self.activity_text.configure(state="disabled")

    def update_dashboard(self):
        if self.df is not None:
            self.card_values["dataset"].config(
                text=f"{len(self.df):,}"
            )

        if self.metrics:
            self.card_values["accuracy"].config(
                text=f"{self.metrics['Accuracy'] * 100:.2f}%"
            )
            self.card_values["precision"].config(
                text=f"{self.metrics['Precision'] * 100:.2f}%"
            )
            self.card_values["recall"].config(
                text=f"{self.metrics['Recall'] * 100:.2f}%"
            )
            self.card_values["f1"].config(
                text=f"{self.metrics['F1 Score'] * 100:.2f}%"
            )
            self.card_values["tp"].config(
                text=f"{self.metrics['TP']:,}"
            )
            self.card_values["tn"].config(
                text=f"{self.metrics['TN']:,}"
            )
            self.card_values["fp"].config(
                text=f"{self.metrics['FP']:,}"
            )
            self.card_values["fn"].config(
                text=f"{self.metrics['FN']:,}"
            )

        if self.label_counts:
            self.card_values["normal"].config(
                text=f"{self.label_counts.get(0, 0):,}"
            )
            self.card_values["attacks"].config(
                text=f"{self.label_counts.get(1, 0):,}"
            )

        if self.selected_features:
            self.card_values["features"].config(
                text=str(len(self.selected_features))
            )

        if self.processed_df is not None:
            self.dashboard_status.config(
                text="MODEL READY — BENIGN/BOTNET classification is ready",
                bg="#e8f5e9",
                fg="#1b5e20"
            )

            for btn in self.dashboard_buttons:
                btn.config(state="normal")

            self.activity_text.configure(state="normal")
            self.activity_text.delete("1.0", "end")
            self.activity_text.insert(
                "end",
                "Dataset loaded\n"
                "Data preprocessing completed\n"
                "Train/test split completed\n"
                "Standard scaling completed\n"
                "RF feature selection completed\n"
                "Voting ensemble trained\n"
                "Evaluation metrics generated\n"
                "Visualization dashboard ready\n"
            )
            self.activity_text.configure(state="disabled")

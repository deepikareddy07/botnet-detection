import tkinter as tk
from tkinter import ttk

from botnet_detector.config import APP_TITLE


class StyleMixin:
    def configure_styles(self):
        self.root.configure(bg="#f3f5fa")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TNotebook",
            background="#f3f5fa",
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            background="#e2e6f2",
            foreground="#2c2f6b",
            padding=(20, 12),
            font=("Segoe UI", 10, "bold")
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", "#3949ab"),
                ("active", "#5c6bc0")
            ],
            foreground=[
                ("selected", "white"),
                ("active", "white")
            ]
        )

        style.configure(
            "TButton",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 7),
            background="#00897b",
            foreground="white",
            borderwidth=0
        )

        style.map(
            "TButton",
            background=[
                ("active", "#00695c"),
                ("disabled", "#c6ccd6")
            ],
            foreground=[
                ("disabled", "#7c8794")
            ]
        )

        style.configure(
            "TLabel",
            background="#f3f5fa",
            foreground="#2c2f45",
            font=("Segoe UI", 10)
        )

        style.configure(
            "TLabelframe",
            background="white",
            bordercolor="#d7dbea"
        )

        style.configure(
            "TLabelframe.Label",
            background="white",
            foreground="#2c2f6b",
            font=("Segoe UI", 10, "bold")
        )

        style.configure(
            "Treeview",
            background="white",
            fieldbackground="white",
            foreground="#2c2f45",
            rowheight=28,
            font=("Segoe UI", 9)
        )

        style.configure(
            "Treeview.Heading",
            background="#3949ab",
            foreground="white",
            font=("Segoe UI", 9, "bold")
        )

        style.map(
            "Treeview",
            background=[("selected", "#c5cae9")],
            foreground=[("selected", "#1a1c3d")]
        )

    def build_header(self):
        header = tk.Frame(
            self.root,
            bg="white",
            height=70,
            highlightbackground="#d7dbea",
            highlightthickness=1
        )
        header.pack(fill="x", padx=10, pady=(10, 5))
        header.pack_propagate(False)

        # Accent strip
        tk.Frame(
            header,
            bg="#00897b",
            width=8
        ).pack(side="left", fill="y")

        inner = tk.Frame(header, bg="white")
        inner.pack(fill="both", expand=True, padx=18, pady=10)

        title_col = tk.Frame(inner, bg="white")
        title_col.pack(side="left", fill="both", expand=True)

        tk.Label(
            title_col,
            text=APP_TITLE,
            bg="white",
            fg="#1a1c3d",
            font=("Segoe UI", 15, "bold"),
            justify="left",
            anchor="w"
        ).pack(anchor="w")

    def build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        self.dashboard_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)

        self.notebook.add(
            self.dashboard_tab,
            text="  Dashboard  "
        )
        self.notebook.add(
            self.train_tab,
            text="  Train & Evaluate  "
        )
        self.notebook.add(
            self.predict_tab,
            text="  Predict CSV  "
        )

        status = tk.Frame(
            self.root,
            bg="#3949ab",
            height=30
        )
        status.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar()
        tk.Label(
            status,
            textvariable=self.status_var,
            bg="#3949ab",
            fg="#e8eaf6",
            anchor="w",
            font=("Segoe UI", 9, "bold")
        ).pack(fill="both", padx=10)

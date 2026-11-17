import threading
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


class PlotsMixin:
    def require_training(self):
        if self.processed_df is None:
            messagebox.showwarning(
                "No Results",
                "Train the model first."
            )
            return False
        return True

    def show_class_distribution(self):
        if not self.require_training():
            return

        labels = [
            "BENIGN",
            "BOTNET"
        ]

        values = [
            self.label_counts.get(0, 0),
            self.label_counts.get(1, 0)
        ]

        plt.close('all')
        plt.figure(
            figsize=(7, 5)
        )

        bars = plt.bar(
            labels,
            values,
            color=[
                "#42a5f5",
                "#ef5350"
            ]
        )

        plt.title(
            "Traffic Class Distribution",
            fontsize=15,
            fontweight="bold"
        )

        plt.ylabel(
            "Number of Flows"
        )

        for bar, value in zip(
            bars,
            values
        ):
            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        plt.grid(
            axis="y",
            alpha=0.18
        )
        plt.tight_layout()
        plt.show(block=False)

    def show_class_pie(self):
        if not self.label_counts:
            return

        values = [
            self.label_counts.get(0, 0),
            self.label_counts.get(1, 0)
        ]

        plt.close('all')
        plt.figure(
            figsize=(7, 6)
        )

        plt.pie(
            values,
            labels=[
                "BENIGN",
                "BOTNET"
            ],
            autopct="%1.1f%%",
            startangle=90,
            colors=[
                "#42a5f5",
                "#ef5350"
            ],
            wedgeprops={
                "width": 0.42,
                "edgecolor": "white"
            },
            textprops={
                "fontsize": 10
            }
        )

        plt.title(
            "Traffic Class Composition",
            fontsize=15,
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show(block=False)

    def show_prediction_pie(self):
        """Pie chart of BENIGN vs BOTNET among the most recent
        Predict-CSV run's results."""
        counts = getattr(self, "_last_prediction_counts", None)
        if not counts or sum(counts) == 0:
            messagebox.showinfo(
                "Not Available",
                "Run a prediction first (Load Traffic CSV & Predict)."
            )
            return

        benign_count, botnet_count = counts

        plt.close('all')
        plt.figure(figsize=(7, 6))
        plt.pie(
            [benign_count, botnet_count],
            labels=["BENIGN", "BOTNET"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#42a5f5", "#ef5350"],
            wedgeprops={"width": 0.42, "edgecolor": "white"},
            textprops={"fontsize": 10}
        )
        plt.title(
            f"Prediction Results ({benign_count + botnet_count:,} flows)",
            fontsize=15,
            fontweight="bold"
        )
        plt.tight_layout()
        plt.show(block=False)

    def show_metrics_chart(self):
        if not self.model_metrics:
            return

        model_names = list(self.model_metrics.keys())
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        x = np.arange(len(model_names))
        width = 0.20

        plt.close("all")
        plt.figure(figsize=(12, 6))

        for i, metric in enumerate(metric_names):
            values = [self.model_metrics[m][metric] for m in model_names]
            bars = plt.bar(x + (i - 1.5) * width, values, width, label=metric)
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    fontsize=8
                )

        plt.xticks(x, model_names, rotation=10)
        plt.ylim(0, 1.08)
        plt.ylabel("Score")
        plt.title("Model Performance Comparison", fontsize=15, fontweight="bold")
        plt.legend()
        plt.grid(axis="y", alpha=0.18)
        plt.tight_layout()
        plt.show(block=False)

    def show_metric_radar(self):
        if not self.model_metrics or "Hard Voting Ensemble" not in self.model_metrics:
            return

        labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
        values = [self.model_metrics["Hard Voting Ensemble"][x] for x in labels]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        values += [values[0]]
        angles = np.concatenate([angles, [angles[0]]])

        plt.close("all")
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=3)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title("Hard Voting Ensemble Performance", pad=25, fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show(block=False)

    def show_confusion_matrix(self):
        if (
            self.y_test is None
            or self.ensemble is None
        ):
            return

        predictions = self.ensemble.predict(
            self.X_test
        )

        cm = confusion_matrix(
            self.y_test,
            predictions
        )

        plt.close('all')
        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        image = ax.imshow(
            cm,
            cmap="Blues"
        )

        ax.set_xticks(
            [0, 1]
        )
        ax.set_yticks(
            [0, 1]
        )

        ax.set_xticklabels(
            ["BENIGN", "BOTNET"]
        )
        ax.set_yticklabels(
            ["BENIGN", "BOTNET"]
        )

        ax.set_xlabel(
            "Predicted Class"
        )
        ax.set_ylabel(
            "Actual Class"
        )

        ax.set_title(
            "Confusion Matrix — Voting Ensemble",
            fontsize=15,
            fontweight="bold"
        )

        threshold = cm.max() * 0.55

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color=(
                        "white"
                        if cm[i, j] > threshold
                        else "#102a43"
                    )
                )

        fig.colorbar(
            image,
            ax=ax
        )

        plt.tight_layout()
        plt.show(block=False)

    def show_model_comparison(self):
        if not self.model_metrics:
            return

        names = list(self.model_metrics.keys())
        scores = [self.model_metrics[n]["Accuracy"] for n in names]

        plt.close("all")
        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, scores)
        plt.ylim(0, 1.05)
        plt.title("Individual Classifiers vs Hard Voting Ensemble", fontsize=15, fontweight="bold")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=10)

        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.015,
                f"{score:.2%}",
                ha="center",
                fontweight="bold"
            )

        plt.grid(axis="y", alpha=0.18)
        plt.tight_layout()
        plt.show(block=False)

    def show_feature_importance(self):
        if not self.feature_importances:
            return

        items = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1]
        )

        names = [
            x[0]
            for x in items
        ]

        values = [
            x[1]
            for x in items
        ]

        plt.close('all')
        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            names,
            values,
            color=plt.cm.viridis(
                np.linspace(
                    0.2,
                    0.9,
                    len(values)
                )
            )
        )

        plt.title(
            "Selected Feature Importance",
            fontsize=15,
            fontweight="bold"
        )

        plt.xlabel(
            "Importance"
        )

        plt.tight_layout()
        plt.show(block=False)

    def show_top_features(self):
        if not self.feature_importances:
            return

        items = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        items.reverse()

        names = [
            x[0]
            for x in items
        ]

        values = [
            x[1]
            for x in items
        ]

        plt.close('all')
        plt.figure(
            figsize=(10, 7)
        )

        bars = plt.barh(
            names,
            values,
            color="#7e57c2"
        )

        plt.title(
            "Top 15 Selected Features",
            fontsize=15,
            fontweight="bold"
        )

        plt.xlabel(
            "Feature Importance"
        )

        for bar, value in zip(
            bars,
            values
        ):
            plt.text(
                value,
                bar.get_y()
                + bar.get_height() / 2,
                f" {value:.3f}",
                va="center",
                fontsize=8
            )

        plt.tight_layout()
        plt.show(block=False)

    def show_attack_type_breakdown(self):
        if not self.raw_label_counts:
            return

        items = sorted(
            self.raw_label_counts.items(),
            key=lambda x: x[1]
        )

        names = [
            x[0]
            for x in items
        ]

        values = [
            x[1]
            for x in items
        ]

        plt.close('all')
        plt.figure(
            figsize=(
                10,
                max(
                    5,
                    len(names) * 0.35
                )
            )
        )

        plt.barh(
            names,
            values,
            color="#ef5350"
        )

        plt.title(
            "Raw Label Distribution",
            fontsize=15,
            fontweight="bold"
        )

        plt.xlabel(
            "Flow Count"
        )

        plt.tight_layout()
        plt.show(block=False)

    def show_pca_scatter(self):
        if self.X_test is None:
            return

        pca = PCA(
            n_components=2,
            random_state=42
        )

        coords = pca.fit_transform(
            self.X_test
        )

        y = np.asarray(
            self.y_test
        )

        plt.close('all')
        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            coords[y == 0, 0],
            coords[y == 0, 1],
            s=10,
            alpha=0.45,
            color="#42a5f5",
            label="BENIGN"
        )

        plt.scatter(
            coords[y == 1, 0],
            coords[y == 1, 1],
            s=10,
            alpha=0.45,
            color="#ef5350",
            label="BOTNET"
        )

        variance = (
            pca.explained_variance_ratio_
        )

        plt.xlabel(
            f"PC1 ({variance[0] * 100:.1f}% variance)"
        )

        plt.ylabel(
            f"PC2 ({variance[1] * 100:.1f}% variance)"
        )

        plt.title(
            "PCA Projection of Selected Traffic Features",
            fontsize=14,
            fontweight="bold"
        )

        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    def show_roc_curve(self):
        if not self.roc_data:
            return

        fpr, tpr, score = (
            self.roc_data
        )

        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(
            fpr, tpr,
            color="#1976d2", linewidth=3, marker="o", markersize=3,
            label=f"Ensemble AUC = {score:.3f}"
        )
        ax.plot(
            [0, 1], [0, 1], "--",
            color="#9e9e9e", label="Chance"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Botnet Detection", fontsize=15, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.18)

        # A curve that shoots straight up then flattens (looking almost like
        # a right angle) means the model separates the classes very cleanly
        # (AUC close to 1) — not a bug. Zoom into the low-FPR corner so that
        # region isn't just a flat line at the top.
        if score > 0.9:
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                axins = inset_axes(ax, width="48%", height="48%", loc="lower right", borderpad=3)
                axins.plot(fpr, tpr, color="#1976d2", linewidth=2)
                axins.plot([0, 1], [0, 1], "--", color="#9e9e9e", linewidth=1)
                nonzero_fpr = fpr[fpr > 0]
                zoom_x = min(0.2, max(float(nonzero_fpr.min()) * 4, 0.02)) if nonzero_fpr.size else 0.1
                mask = fpr <= zoom_x
                y_low = max(0.0, float(tpr[mask].min()) - 0.03) if mask.any() else 0.8
                axins.set_xlim(0, zoom_x)
                axins.set_ylim(y_low, 1.01)
                axins.grid(alpha=0.3)
                axins.set_title("Zoomed near origin", fontsize=8)
                axins.tick_params(labelsize=7)
            except Exception:
                pass  # inset is a nice-to-have; skip quietly if it fails

        fig.tight_layout()
        plt.show(block=False)

    def show_pr_curve(self):
        if not self.pr_data:
            return

        precision, recall, ap = (
            self.pr_data
        )

        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(
            recall, precision,
            color="#8e24aa", linewidth=3, marker="o", markersize=3,
            label=f"Average Precision = {ap:.3f}"
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.18)

        # With very few positive (attack) examples in the test set, Recall
        # often stays pinned at 1.0 across most thresholds while Precision
        # climbs — that draws as one long vertical line, with the real
        # curve detail compressed into a small corner near the tail. Zoom
        # into that region so it isn't just a straight segment.
        if ap > 0.85:
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                axins = inset_axes(ax, width="48%", height="48%", loc="lower right", borderpad=3)
                axins.plot(recall, precision, color="#8e24aa", linewidth=2, marker="o", markersize=3)
                recall_below_1 = recall[recall < 1.0]
                zoom_x_low = max(0.0, float(recall_below_1.max()) - 0.25) if recall_below_1.size else 0.7
                axins.set_xlim(zoom_x_low, 1.02)
                axins.set_ylim(0.0, 1.05)
                axins.grid(alpha=0.3)
                axins.set_title("Zoomed near Recall = 1", fontsize=8)
                axins.tick_params(labelsize=7)
            except Exception:
                pass  # inset is a nice-to-have; skip quietly if it fails

        fig.tight_layout()
        plt.show(block=False)

    def start_learning_curve(self):
        """Learning curve retrains the ensemble across several training-set
        sizes with cross-validation — that's inherently slow (RF + LinearSVC
        + LR fit ~15 times), so it now runs on a background thread instead
        of blocking the window (which is what made it look 'hung')."""
        if self.ensemble is None or self.X_train is None:
            messagebox.showwarning(
                "No Model",
                "Train the model first (Preprocess + Train Ensemble)."
            )
            return
        if self._busy:
            messagebox.showinfo(
                "Busy",
                "Another background task is still running. Please wait."
            )
            return

        proceed = messagebox.askyesno(
            "Learning Curve",
            "This retrains the ensemble across 5 training-set sizes with "
            "3-fold cross-validation (~15 fits total). It can take a while "
            "on larger datasets. Continue?"
        )
        if not proceed:
            return

        self._busy = True
        self.progress.start(12)
        self.status_var.set("Computing learning curve (this may take a while)...")

        threading.Thread(target=self._lc_worker, daemon=True).start()

    def _lc_worker(self):
        try:
            model = clone(self.ensemble)

            sizes, train_scores, val_scores = learning_curve(
                model, self.X_train, self.y_train,
                cv=3, train_sizes=np.linspace(0.2, 1.0, 5),
                scoring="accuracy", n_jobs=1
            )

            train_mean = train_scores.mean(axis=1)
            val_mean = val_scores.mean(axis=1)

            # Very small training subsets can leave a CV fold with too few
            # minority-class (attack) examples to fit at all; sklearn then
            # silently records NaN for that point instead of raising. Drop
            # any such points so the chart doesn't quietly lose data, and
            # say so in the log rather than staying silent about it.
            valid = ~(np.isnan(train_mean) | np.isnan(val_mean))
            if not valid.all():
                dropped = int((~valid).sum())
                self.msg_queue.put((
                    "log",
                    f"Note: {dropped} training-size point(s) were skipped — "
                    "too few BOTNET samples in that subset for stable "
                    "3-fold cross-validation. This is a small-sample effect "
                    "from class imbalance, not an error."
                ))
            sizes, train_mean, val_mean = sizes[valid], train_mean[valid], val_mean[valid]

            self.msg_queue.put((
                "log",
                "\n--- LEARNING CURVE ---\n"
                f"Train sizes: {list(sizes)}\n"
                f"Training accuracy  : {[round(v, 4) for v in train_mean]}\n"
                f"Validation accuracy: {[round(v, 4) for v in val_mean]}"
            ))
            self.msg_queue.put(("lc_done", (sizes, train_mean, val_mean)))
        except Exception as e:
            self.msg_queue.put(("lc_error", str(e)))

    def _draw_learning_curve(self, sizes, train_mean, val_mean):
        plt.close('all')
        plt.figure(figsize=(8, 5))

        plt.plot(
            sizes, train_mean, "o-",
            color="#1976d2", linewidth=2, label="Training Accuracy"
        )
        plt.plot(
            sizes, val_mean, "o-",
            color="#ef6c00", linewidth=2, label="Validation Accuracy"
        )

        plt.title("Learning Curve — Voting Ensemble", fontsize=15, fontweight="bold")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.18)
        plt.tight_layout()
        plt.show(block=False)

    def show_learning_curve(self):
        # kept as an alias so any existing button wiring keeps working
        self.start_learning_curve()

    def find_column(self, candidates):
        if self.processed_df is None:
            return None

        lower_map = {
            str(c).lower(): c
            for c in self.processed_df.columns
        }

        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[
                    candidate.lower()
                ]

        for col in self.processed_df.columns:
            for candidate in candidates:
                if candidate.lower() in str(
                    col
                ).lower():
                    return col

        return None

    def show_duration_distribution(self):
        if (
            self.processed_df is None
            or "Flow Duration"
            not in self.processed_df.columns
        ):
            messagebox.showinfo(
                "Not Available",
                "Flow Duration column was not found."
            )
            return

        benign = self.processed_df.loc[
            self.processed_df["Label"] == 0,
            "Flow Duration"
        ]

        botnet = self.processed_df.loc[
            self.processed_df["Label"] == 1,
            "Flow Duration"
        ]

        plt.close('all')
        plt.figure(
            figsize=(9, 5)
        )

        bins = np.histogram_bin_edges(
            pd.concat(
                [benign, botnet]
            ),
            bins=40
        )

        plt.hist(
            benign,
            bins=bins,
            alpha=0.65,
            color="#42a5f5",
            label="BENIGN"
        )

        plt.hist(
            botnet,
            bins=bins,
            alpha=0.65,
            color="#ef5350",
            label="BOTNET"
        )

        plt.yscale(
            "log"
        )

        plt.title(
            "Flow Duration Distribution by Class",
            fontsize=15,
            fontweight="bold"
        )

        plt.xlabel(
            "Flow Duration"
        )

        plt.ylabel(
            "Count (log scale)"
        )

        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    def show_correlation_heatmap(self):
        if not self.selected_features:
            return

        corr = self.processed_df[
            self.selected_features
        ].corr()

        plt.close('all')
        plt.figure(
            figsize=(10, 8)
        )

        plt.imshow(
            corr,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        plt.colorbar()

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=90,
            fontsize=7
        )

        plt.yticks(
            range(len(corr.columns)),
            corr.columns,
            fontsize=7
        )

        plt.title(
            "Selected Feature Correlation Heatmap",
            fontsize=15,
            fontweight="bold"
        )

        plt.tight_layout()
        plt.show(block=False)

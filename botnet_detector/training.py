import threading
from tkinter import messagebox

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class TrainingMixin:
    def start_training(self):
        if self.df is None:
            messagebox.showwarning(
                "No Dataset",
                "Load a dataset first."
            )
            return

        if self._busy:
            messagebox.showinfo(
                "Busy",
                "Another background task is still running. Please wait."
            )
            return

        self._busy = True
        self.train_btn.config(state="disabled")
        self.load_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Training in progress...")

        threading.Thread(target=self._train_worker, daemon=True).start()

    @staticmethod
    def _find_column_in(df, candidates):
        """Fuzzy column lookup usable on any dataframe directly (unlike
        find_column(), which only looks at self.processed_df)."""
        lower_map = {str(c).lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        for col in df.columns:
            for candidate in candidates:
                if candidate.lower() in str(col).lower():
                    return col
        return None

    def _apply_initial_traffic_filtering(self, df):
        """The 'Initial Traffic Filtering' stage from the architecture
        diagram. Runs AFTER Data Preprocessing and BEFORE Feature
        Selection, per the current pipeline order.

        Data Preprocessing already handles generic ML data hygiene
        (duplicates, unparseable labels) — that part of the pipeline has
        nothing to do with networking. This stage is different on purpose:
        it applies cheap, network-domain-specific validity checks that only
        make sense for traffic data, using every packet each flow actually
        has (no packet-count window or cap):

          1. Protocol validity — the IP protocol field is a single byte
             (0-255). Any row outside that range is a corrupted/malformed
             flow record, not real traffic.
          2. Port validity — TCP/UDP ports are 16-bit (0-65535). Same idea:
             out-of-range values indicate parsing/export errors upstream.
          3. Rate sanity — Flow Bytes/s and Flow Packets/s should follow a
             roughly consistent distribution for real traffic. Extreme
             statistical outliers (beyond 3x the IQR) are flagged as
             implausible/corrupted rate profiles using Tukey's method, a
             standard, data-driven outlier rule — not an arbitrary cutoff.

        None of these are things "remove duplicate rows" or "normalize
        columns" would ever catch — they require knowing what a protocol
        number or a port or a packet rate actually means.
        """
        total_rows = len(df)

        proto_col = self._find_column_in(df, ["Protocol"])
        if proto_col is not None:
            before = len(df)
            df = df[df[proto_col].between(0, 255)]
            self.msg_queue.put((
                "log",
                f"Initial traffic filtering: dropped flows with invalid IP "
                f"protocol values (using '{proto_col}', valid range 0-255) — "
                f"{before} -> {len(df)} flows"
            ))

        port_cols = [
            c for c in (
                self._find_column_in(df, ["Destination Port"]),
                self._find_column_in(df, ["Source Port"]),
            ) if c is not None
        ]
        for port_col in port_cols:
            before = len(df)
            df = df[df[port_col].between(0, 65535)]
            self.msg_queue.put((
                "log",
                f"Initial traffic filtering: dropped flows with invalid "
                f"port values (using '{port_col}', valid range 0-65535) — "
                f"{before} -> {len(df)} flows"
            ))

        rate_cols = [
            c for c in (
                self._find_column_in(df, ["Flow Bytes/s"]),
                self._find_column_in(df, ["Flow Packets/s"]),
            ) if c is not None
        ]
        for rate_col in rate_cols:
            before = len(df)
            q1, q3 = df[rate_col].quantile(0.25), df[rate_col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                upper = q3 + 3 * iqr
                df = df[df[rate_col] <= upper]
                self.msg_queue.put((
                    "log",
                    f"Initial traffic filtering: dropped flows with an "
                    f"implausible rate profile (using '{rate_col}', Tukey's "
                    f"3xIQR rule, cutoff={upper:.1f}) — {before} -> {len(df)} flows"
                ))

        self.msg_queue.put((
            "log",
            f"Initial traffic filtering complete — {total_rows} -> {len(df)} "
            f"flows retained (protocol/port/rate validity, full flow data used)"
        ))

        return df

    def _train_worker(self):
        """Runs the full preprocess -> feature selection -> ensemble
        training -> evaluation pipeline on a background thread, so the
        window stays responsive. Never touch Tkinter widgets from here —
        only put messages on self.msg_queue; the main thread applies them
        in _poll_queue."""
        try:
            df = self.df.copy()

            self.msg_queue.put(("log", "\n--- PREPROCESSING ---"))
            df = df.drop_duplicates()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            if "Label" not in df.columns:
                raise ValueError("The selected dataset must contain a 'Label' column.")

            raw_label_counts = df["Label"].astype(str).str.strip().value_counts().to_dict()

            def map_label(value):
                text = str(value).strip().upper()
                benign_labels = {
                    "BENIGN", "NORMAL", "NORMAL TRAFFIC",
                    "BACKGROUND", "BACKGROUND TRAFFIC", "0"
                }
                return 0 if text in benign_labels or text.startswith("NORMAL") else 1

            df["Label"] = df["Label"].apply(map_label)

            # Handle invalid values during preprocessing.
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["Label"], inplace=True)
            df.drop_duplicates(inplace=True)

            # Basic numeric sanity check — belongs here, not in Initial
            # Traffic Filtering, since it's generic data hygiene (an
            # implausible negative/zero value) rather than a networking-
            # domain judgment about protocols, ports, or traffic rates.
            dur_col = self._find_column_in(df, ["Flow Duration"])
            if dur_col is not None:
                before_dur = len(df)
                df = df[df[dur_col] > 0]
                self.msg_queue.put((
                    "log",
                    f"Preprocessing: dropped rows with non-positive "
                    f"'{dur_col}' — {before_dur} -> {len(df)} rows"
                ))

            label_counts = df["Label"].value_counts().to_dict()

            self.msg_queue.put((
                "log",
                f"Class counts: BENIGN={label_counts.get(0, 0)}, BOTNET={label_counts.get(1, 0)}"
            ))

            if df["Label"].nunique() < 2:
                raise ValueError("Both BENIGN and BOTNET classes are required for training.")

            self.msg_queue.put(("log", "\n--- INITIAL TRAFFIC FILTERING ---"))
            df = self._apply_initial_traffic_filtering(df)

            processed_df = df.copy()

            X = df.drop("Label", axis=1)
            y = df["Label"]

            feature_columns = list(X.columns)

            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                self.msg_queue.put(("log", f"Dropping non-numeric flow columns: {non_numeric}"))
                X = X.drop(columns=non_numeric)
                feature_columns = list(X.columns)

            # Remove columns that are completely invalid. For remaining missing
            # values, imputation is fitted separately on the training data below.
            all_nan_columns = X.columns[X.isna().all()].tolist()
            if all_nan_columns:
                X = X.drop(columns=all_nan_columns)
                feature_columns = list(X.columns)
                self.msg_queue.put(("log", f"Removed all-NaN columns: {all_nan_columns}"))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

            # Median imputation is fitted only on training data to prevent leakage.
            train_medians = X_train.median(numeric_only=True)
            X_train = X_train.fillna(train_medians).fillna(0)
            X_test = X_test.fillna(train_medians).fillna(0)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            self.msg_queue.put((
                "log", f"Train/test split: {X_train_s.shape} / {X_test_s.shape}"
            ))

            self.msg_queue.put(("log", "\n--- FEATURE SELECTION ---"))

            selector = SelectFromModel(
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced"
                ),
                threshold="median"
            )
            selector.fit(X_train_s, y_train)

            X_train_sel = selector.transform(X_train_s)
            X_test_sel = selector.transform(X_test_s)

            selected_features = list(X.columns[selector.get_support()])

            importances = selector.estimator_.feature_importances_[selector.get_support()]
            feature_importances = dict(zip(selected_features, importances))

            self.msg_queue.put(("log", "Selected features:"))
            self.msg_queue.put(("log", ", ".join(selected_features)))

            self.msg_queue.put(("log", "\n--- TRAINING ENSEMBLE ---"))

            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
            svm = LinearSVC(
                C=1.0,
                random_state=42,
                max_iter=5000,
                class_weight="balanced"
            )
            lr = LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight="balanced"
            )

            # Weighted hard voting: unweighted voting lets any two of three
            # classifiers always outvote the third, regardless of which is
            # actually more reliable. Empirically, Random Forest is the
            # best-calibrated of the three on this heavily imbalanced
            # traffic (see Model Comparison), while Linear SVM and Logistic
            # Regression — both linear, both reacting the same way to
            # class_weight="balanced" under severe imbalance — tend to vote
            # together and can outvote a correct Random Forest prediction.
            # Weighting Random Forest at 3 versus 1 each for SVM/LR means
            # Random Forest alone can outweigh SVM+LR agreeing with each
            # other, instead of always losing 1-vs-2.
            ensemble = VotingClassifier(
                estimators=[("rf", rf), ("svm", svm), ("lr", lr)],
                voting="hard",
                weights=[3, 1, 1]
            )
            ensemble.fit(X_train_sel, y_train)

            # Comparison-only addition: Gradient Boosting is NOT part of the
            # voting ensemble (the ensemble stays exactly RF + SVM + LR, per
            # the title) — it's trained separately purely to show in the
            # Model Comparison output how a sequential boosting approach
            # stacks up against this project's parallel-vote architecture.
            gb = GradientBoostingClassifier(random_state=42)
            gb.fit(X_train_sel, y_train)
            gb_pred = gb.predict(X_test_sel)

            # Individual classifier predictions for the required comparison.
            rf_pred = ensemble.named_estimators_["rf"].predict(X_test_sel)
            svm_pred = ensemble.named_estimators_["svm"].predict(X_test_sel)
            lr_pred = ensemble.named_estimators_["lr"].predict(X_test_sel)
            y_pred = ensemble.predict(X_test_sel)

            def model_scores(y_true, pred):
                return {
                    "Accuracy": accuracy_score(y_true, pred),
                    "Precision": precision_score(y_true, pred, zero_division=0),
                    "Recall": recall_score(y_true, pred, zero_division=0),
                    "F1 Score": f1_score(y_true, pred, zero_division=0),
                }

            model_metrics = {
                "Random Forest": model_scores(y_test, rf_pred),
                "Linear SVM": model_scores(y_test, svm_pred),
                "Logistic Regression": model_scores(y_test, lr_pred),
                "Gradient Boosting (comparison only)": model_scores(y_test, gb_pred),
                "Hard Voting Ensemble": model_scores(y_test, y_pred),
            }

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            metrics = {
                **model_metrics["Hard Voting Ensemble"],
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            }

            report = classification_report(
                y_test, y_pred,
                labels=[0, 1],
                target_names=["BENIGN", "BOTNET"],
                zero_division=0
            )

            self.msg_queue.put(("log", "\n===== INDIVIDUAL MODEL vs ENSEMBLE ====="))
            for model_name, values in model_metrics.items():
                self.msg_queue.put((
                    "log",
                    f"{model_name:<24} Accuracy={values['Accuracy']:.4f}  "
                    f"Precision={values['Precision']:.4f}  "
                    f"Recall={values['Recall']:.4f}  F1={values['F1 Score']:.4f}"
                ))

            self.msg_queue.put(("log", "\n===== HARD VOTING ENSEMBLE EVALUATION ====="))
            for name, value in metrics.items():
                if name in {"TP", "TN", "FP", "FN"}:
                    self.msg_queue.put(("log", f"{name:<10}: {int(value)}"))
                else:
                    self.msg_queue.put(("log", f"{name:<10}: {value:.4f} ({value * 100:.2f}%)"))
            self.msg_queue.put(("log", f"\nClassification Report:\n{report}"))

            # ROC/PR score for the hard-voting decision: proportion of the
            # three base classifiers voting BOTNET (0, 1/3, 2/3, or 1).
            vote_score = (
                (rf_pred == 1).astype(float) +
                (svm_pred == 1).astype(float) +
                (lr_pred == 1).astype(float)
            ) / 3.0

            fpr, tpr, _ = roc_curve(y_test, vote_score)
            roc_auc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(y_test, vote_score)
            ap = average_precision_score(y_test, vote_score)

            self.msg_queue.put(("log", f"\nVoting ROC AUC: {roc_auc:.4f}"))
            self.msg_queue.put(("log", f"Average Precision: {ap:.4f}"))
            if roc_auc > 0.97:
                self.msg_queue.put((
                    "log",
                    "Note: an AUC this high makes the ROC curve look like a "
                    "near-right angle (shoots up, then flat) instead of a "
                    "smooth diagonal-ish curve — that's expected for a strong "
                    "classifier, not a bug. The ROC chart adds a zoomed-in "
                    "inset near the origin so its actual shape is visible."
                ))
            n_botnet_test = int((y_test == 1).sum())
            if ap > 0.85 and n_botnet_test < 100:
                self.msg_queue.put((
                    "log",
                    f"Note: only {n_botnet_test} BOTNET samples are in the "
                    "test split, so Recall stays at 1.0 across most decision "
                    "thresholds — the Precision-Recall curve draws mostly as "
                    "a straight vertical line, with the real detail compressed "
                    "into a small corner. This is a small-sample effect, not "
                    "a bug; the PR chart adds a zoomed-in inset near "
                    "Recall = 1 to show that region clearly."
                ))

            bundle = {
                "df_len": len(self.df),
                "processed_df": processed_df,
                "scaler": scaler,
                "selector": selector,
                "ensemble": ensemble,
                "selected_features": selected_features,
                "feature_columns": feature_columns,
                "train_medians": train_medians.to_dict(),
                "X_train": X_train_sel, "X_test": X_test_sel,
                "y_train": y_train, "y_test": y_test,
                "metrics": metrics,
                "model_metrics": model_metrics,
                "feature_importances": feature_importances,
                "label_counts": label_counts,
                "raw_label_counts": raw_label_counts,
                "roc_data": (fpr, tpr, roc_auc),
                "pr_data": (precision, recall, ap),
            }
            self.msg_queue.put(("train_done", bundle))
        except Exception as e:
            self.msg_queue.put(("train_error", str(e)))

    def _apply_training_result(self, bundle):
        self.processed_df = bundle["processed_df"]
        self.scaler = bundle["scaler"]
        self.selector = bundle["selector"]
        self.ensemble = bundle["ensemble"]
        self.selected_features = bundle["selected_features"]
        self.feature_columns = bundle["feature_columns"]
        self.train_medians = bundle.get("train_medians", {})
        self.X_train, self.X_test = bundle["X_train"], bundle["X_test"]
        self.y_train, self.y_test = bundle["y_train"], bundle["y_test"]
        self.metrics = bundle["metrics"]
        self.model_metrics = bundle.get("model_metrics", {})
        self.feature_importances = bundle["feature_importances"]
        self.label_counts = bundle["label_counts"]
        self.raw_label_counts = bundle["raw_label_counts"]
        self.roc_data = bundle["roc_data"]
        self.pr_data = bundle["pr_data"]

        self.update_dashboard()
        self.enable_visualizations()

        self._busy = False
        self.progress.stop()
        self.load_btn.config(state="normal")
        self.train_btn.config(state="normal")
        self.save_model_btn.config(state="normal")

        self.status_var.set("Training completed successfully.")
        messagebox.showinfo(
            "Training Complete",
            "The botnet detection ensemble was trained successfully."
        )

    def enable_visualizations(self):
        """Turns on every visualization button (Train tab + Dashboard tab).
        Call this once real results exist to plot (after training, or after
        loading a saved model bundle)."""
        for btn in getattr(self, "graph_buttons", []):
            btn.config(state="normal")
        for btn in getattr(self, "dashboard_buttons", []):
            btn.config(state="normal")

# Ensemble Machine Learning Based Early Botnet Detection Using Initial Traffic Filtering

A desktop application that classifies network flows as BENIGN or BOTNET using a weighted hard-voting ensemble of Random Forest, Linear SVM and Logistic Regression. Flows are reconstructed directly from a raw CTU-13 PCAP capture, labelled using the known infected host IP of the chosen scenario. A labelled flow CSV can also be loaded instead.

## Pipeline

1. Data preprocessing: duplicate removal, infinite and missing value handling, label mapping to BENIGN (0) and BOTNET (1), removal of non-positive flow durations.
2. Initial traffic filtering: protocol range check (0-255), port range check (0-65535), and rate outlier removal (Tukey 3x IQR on Flow Bytes/s and Flow Packets/s).
3. Train/test split: 80/20, stratified.
4. Median imputation and standard scaling, fitted on the training split only.
5. Feature selection: Random Forest importance with a median threshold.
6. Ensemble: hard voting over Random Forest (weight 3), Linear SVM (weight 1) and Logistic Regression (weight 1), all with balanced class weights.
7. Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC, precision-recall and learning curves. Gradient Boosting is trained only for comparison and is not part of the ensemble.

For PCAP input, an early-detection window can be set so that flow features are computed from only the first N packets of each flow.

## Project structure

```
main.py                     application entry point
data/                       place the downloaded CTU-13 PCAP here
requirements.txt
botnet_detector/
    config.py               application title
    dataset.py              CSV dataset loading
    pcap.py                 PCAP reading, flow reconstruction, infected-host auto-locate
    training.py             preprocessing, initial traffic filtering, training, evaluation
    prediction.py           prediction on new CSV or PCAP flows, saving results
    model_io.py             save and load trained model bundles
    plots.py                matplotlib visualizations
    gui/
        app.py              main window, combines all components
        styles.py           ttk styles, header, notebook, status bar
        dashboard.py        dashboard tab
        train_tab.py        train and evaluate tab
        predict_tab.py      prediction tab
```

## Installation

Python 3.9 or newer is required. Tkinter ships with the standard Python installer on Windows and macOS. On Debian or Ubuntu install it with `sudo apt install python3-tk`.

```
git clone https://github.com/deepikareddy07/botnet-detection.git
cd botnet-detection
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Usage

```
python main.py
```

1. Train and Evaluate tab: click Load PCAP File, enter the infected IP and settings, then click Preprocess + Train Ensemble. A labelled CSV with a `Label` column can be loaded instead.
2. Use the visualization buttons to inspect results, and Save Model to store the trained bundle as a `.pkl` file.
3. Predict CSV tab: load a saved model, then classify a new traffic CSV or PCAP and save the predictions.

## Dataset

This project uses the CTU-13 botnet captures (Stratosphere Lab, CTU Prague). The capture file is about 1.5 GB, so it is not stored in the repository itself. Download it from the Releases page of this repository (https://github.com/deepikareddy07/botnet-detection/releases), or get the original from the CTU-13 page, and place it in the `data/` folder.

1. Train and Evaluate tab, click Load PCAP File and select the capture. Compressed `.pcap.bz2` and `.pcap.gz` files are read directly.
2. In the settings dialog, enter the infected host IP of that scenario. The field is pre-filled with 147.32.84.165.
3. Keep auto-locate enabled so the loader finds where the infected host's traffic starts in the capture.
4. Set the early detection window (number of packets per flow) and click Load PCAP, then Preprocess + Train Ensemble.

## Notes

- Saved `.pkl` model files should only be loaded if you created them yourself, since pickle files can execute code when loaded.

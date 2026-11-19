import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

def download_and_extract(extract_dir="UCI HAR Dataset"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = "UCI_HAR_Dataset.zip"
    if not os.path.exists(extract_dir):
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

def load_signals(subset, dataset_path="UCI HAR Dataset/"):
    signals_data = []
    for signal in SIGNALS:
        filename = f"{dataset_path}{subset}/Inertial Signals/{signal}_{subset}.txt"
        data = pd.read_csv(filename, sep=r"\s+", header=None).values
        signals_data.append(data)
    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset, dataset_path="UCI HAR Dataset/"):
    filename = f"{dataset_path}{subset}/y_{subset}.txt"
    return pd.read_csv(filename, sep=r"\s+", header=None)[0].values - 1

def load_subjects(subset, dataset_path="UCI HAR Dataset/"):
    filename = f"{dataset_path}{subset}/subject_{subset}.txt"
    return pd.read_csv(filename, sep=r"\s+", header=None)[0].values

def extract_custom_features(X):
    N, T, C = X.shape
    features = []
    for c in range(C):
        sig = X[:, :, c]
        features.extend([
            np.mean(sig, axis=1),
            np.std(sig, axis=1),
            np.sqrt(np.mean(sig**2, axis=1)),
            np.sum(sig**2, axis=1) / T
        ])
    body_acc_sma = np.sum(np.abs(X[:, :, 0]) + np.abs(X[:, :, 1]) + np.abs(X[:, :, 2]), axis=1) / T
    body_gyro_sma = np.sum(np.abs(X[:, :, 3]) + np.abs(X[:, :, 4]) + np.abs(X[:, :, 5]), axis=1) / T
    features.extend([body_acc_sma, body_gyro_sma])

    total_acc_mag = np.sqrt(X[:, :, 6]**2 + X[:, :, 7]**2 + X[:, :, 8]**2)
    features.extend([np.mean(total_acc_mag, axis=1), np.std(total_acc_mag, axis=1)])

    for c1, c2 in [(0, 1), (0, 2), (3, 4)]:
        corr = np.array([
            np.corrcoef(X[i, :, c1], X[i, :, c2])[0, 1]
            if (np.std(X[i, :, c1]) > 1e-6 and np.std(X[i, :, c2]) > 1e-6) else 0.0
            for i in range(N)
        ])
        features.append(corr)

    freqs = rfftfreq(T, d=1/50.0)
    for c in [0, 3]:
        fft_vals = np.abs(rfft(X[:, :, c], axis=1))
        dom_freq = freqs[1:][np.argmax(fft_vals[:, 1:], axis=1)]
        spec_entropy = np.array([entropy(fft_vals[i, 1:] + 1e-12) for i in range(N)])
        features.extend([dom_freq, spec_entropy])

    return np.column_stack(features)

def prepare_datasets():
    download_and_extract()
    X_train_raw, y_train, sub_train = load_signals("train"), load_y("train"), load_subjects("train")
    X_te_raw, y_te = load_signals("test"), load_y("test")

    val_mask = np.isin(sub_train, [1, 3, 5, 6])
    X_tr_raw, y_tr = X_train_raw[~val_mask], y_train[~val_mask]
    X_val_raw, y_val = X_train_raw[val_mask], y_train[val_mask]

    X_tr_feat = extract_custom_features(X_tr_raw)
    X_val_feat = extract_custom_features(X_val_raw)
    X_te_feat = extract_custom_features(X_te_raw)

    raw_scaler = StandardScaler()
    X_tr_raw_s = raw_scaler.fit_transform(X_tr_raw.reshape(-1, 9)).reshape(X_tr_raw.shape)
    X_val_raw_s = raw_scaler.transform(X_val_raw.reshape(-1, 9)).reshape(X_val_raw.shape)
    X_te_raw_s = raw_scaler.transform(X_te_raw.reshape(-1, 9)).reshape(X_te_raw.shape)

    feat_scaler = StandardScaler()
    X_tr_feat_s = feat_scaler.fit_transform(X_tr_feat)
    X_val_feat_s = feat_scaler.transform(X_val_feat)
    X_te_feat_s = feat_scaler.transform(X_te_feat)

    return (X_tr_raw_s, X_val_raw_s, X_te_raw_s,
            X_tr_feat_s, X_val_feat_s, X_te_feat_s,
            y_tr, y_val, y_te, raw_scaler)

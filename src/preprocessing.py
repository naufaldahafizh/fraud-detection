# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """
    Load data dari CSV dan menghapus baris dengan missing value.
    """
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df

def scale_features(X, columns_to_scale=None):
    """
    Menormalisasi fitur numerik menggunakan StandardScaler.
    """
    scaler = StandardScaler()
    if columns_to_scale is None:
        columns_to_scale = X.columns
    X_scaled = X.copy()
    X_scaled[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
    return X_scaled, scaler

def handle_imbalance(X, y):
    """
    Menyeimbangkan kelas dengan SMOTE.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

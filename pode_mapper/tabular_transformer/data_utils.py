# data_utils.py - Data preprocessing utilities for Tabular Transformer

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ============================================================
# Default feature list (32 clinical indicators + FundusAge)
# Matches columns in full_age_02.xlsx / predictions_with_deltas.xlsx
# ============================================================
DEFAULT_FEATURE_NAMES = [
    'BMI', 'SBP', 'DBP', 'AS_level', 'MCHC', 'Creatinine',
    'WBC', 'PDW', 'HDL-C', 'FBG', 'Lymphocyte_Count',
    'Neutrophil_Count', 'LDL-C', 'MCH', 'RDW-CV', 'RBC',
    'Eosinophil_Count', 'PCT', 'MPV', 'PLT', 'HCT',
    'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV',
    'TG', 'TC', 'HGB', 'UA', 'Urine_pH', 'USG',
    'FundusAge'    # Predicted_Age from pode_base (renamed)
]

# Target column (what the model predicts)
TARGET_COLUMN = 'Age'


def load_data(file_path: str, feature_names: list = None, target_col: str = TARGET_COLUMN):
    """
    Load and preprocess the combined clinical + FundusAge table.

    Expected input file columns:
        - All clinical indicators (BMI, SBP, DBP, ..., USG)
        - 'FundusAge' or 'Predicted_Age' (age predicted by pode_base)
        - 'Age' (true chronological age)

    Args:
        file_path: path to xlsx or csv file
        feature_names: list of feature column names (default: DEFAULT_FEATURE_NAMES)
        target_col: target column name (default: 'Age')

    Returns:
        X: pd.DataFrame with feature columns
        y: pd.Series with target values
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    print(f"Loading data from {file_path}...")
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Handle 'FundusAge' alias — might be stored as 'Predicted_Age' or 'Base_Age'
    if 'FundusAge' not in df.columns:
        if 'Predicted_Age' in df.columns:
            df = df.rename(columns={'Predicted_Age': 'FundusAge'})
            print("Renamed 'Predicted_Age' → 'FundusAge'")
        elif 'Base_Age' in df.columns:
            df = df.rename(columns={'Base_Age': 'FundusAge'})
            print("Renamed 'Base_Age' → 'FundusAge'")
        else:
            raise ValueError(
                "Cannot find FundusAge column. Expected one of: "
                "'FundusAge', 'Predicted_Age', 'Base_Age'"
            )

    # Check for missing columns
    available = set(df.columns)
    missing = [f for f in feature_names if f not in available]
    if missing:
        print(f"Warning: {len(missing)} feature columns not found, will be filled with NaN: {missing}")
        for col in missing:
            df[col] = np.nan

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    X = df[feature_names].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    print(f"Valid samples after dropping NaN targets: {len(X)}")
    return X, y


def preprocess_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    scaler_path: str = None
):
    """
    Fit StandardScaler on train, transform all splits.
    Also fills NaN with median of training set.

    Args:
        X_train, X_val, X_test: feature DataFrames
        scaler_path: if given, save fitted scaler to this path

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled (numpy arrays)
        scaler: fitted StandardScaler
    """
    # Fill NaN with column medians (computed from train)
    medians = X_train.median()
    X_train_filled = X_train.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled).astype(np.float32)

    results = [X_train_scaled]

    for X in [X_val, X_test]:
        if X is not None:
            X_filled = X.fillna(medians)
            X_scaled = scaler.transform(X_filled).astype(np.float32)
            results.append(X_scaled)
        else:
            results.append(None)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'medians': medians.to_dict()}, f)
        print(f"Scaler saved to {scaler_path}")

    return results[0], results[1], results[2], scaler


def load_scaler(scaler_path: str):
    """Load a previously saved scaler."""
    with open(scaler_path, 'rb') as f:
        data = pickle.load(f)
    return data['scaler'], pd.Series(data['medians'])

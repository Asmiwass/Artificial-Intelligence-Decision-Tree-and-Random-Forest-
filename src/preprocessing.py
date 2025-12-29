import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


# Paths

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# Dataset loaders

def load_penguins_data(filename="palmer-penguins.csv"):
    """
    Load and preprocess the Palmer Penguins dataset.
    """
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in data/. "
            "Please place the file in the data/ directory."
        )

    df = pd.read_csv(path)

    # Penguins dataset contains missing values
    df = df.dropna().reset_index(drop=True)

    y = df["species"]
    X = df.drop(columns=["species"])

    return X, y


def load_thyroid_data(filename="Thyroid_Diff.csv"):
    """
    Load and preprocess the Thyroid Cancer Recurrence dataset.
    """
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in data/. "
            "Please place the file in the data/ directory."
        )

    df = pd.read_csv(path)

    y = df["Recurred"]
    X = df.drop(columns=["Recurred"])

    return X, y


# Preprocessing utilities

def encode_categorical_features(X):
    """
    Encode categorical columns using Ordinal Encoding.
    """
    X_enc = X.copy()
    categorical_cols = X_enc.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X_enc[categorical_cols] = encoder.fit_transform(
            X_enc[categorical_cols]
        )

    return X_enc


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


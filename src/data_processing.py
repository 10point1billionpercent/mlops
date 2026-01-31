import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from helper_functions import log_info, log_error

# Base directory (mlops/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env explicitly
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Artifacts directory
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv("ARTIFACTS_DIR", "artifacts"))
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Artifact paths
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")


def create_data_pipeline(data: pd.DataFrame):
    """
    Creates preprocessing pipeline with OHE + MinMaxScaler
    """
    categorical_features = data.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

    transformers = []

    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        )

    if numerical_features:
        transformers.append(
            ("num", MinMaxScaler(), numerical_features)
        )

    if not transformers:
        log_error("No categorical or numerical features found.")
        return None

    pipeline = Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers))
        ]
    )

    log_info("Data processing pipeline created successfully.")
    return pipeline


def save_pipeline(pipeline):
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    log_info(f"Pipeline saved at {PIPELINE_PATH}")


def encode_response_variable(y: pd.Series):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    log_info(f"Label encoder saved at {LABEL_ENCODER_PATH}")
    return y_encoded


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


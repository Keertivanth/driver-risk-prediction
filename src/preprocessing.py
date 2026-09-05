"""Data preprocessing utilities."""
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_data(file_path):
    """
    Load the driver risk dataset.
    """

    df = pd.read_csv(
        file_path,
        sep="\t"
    )

    return df


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """
    Clean column names by removing spaces
    and replacing them with underscores.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


# ============================================================
# 3. CLEAN DATASET
# ============================================================

def clean_data(df):
    """
    Perform basic dataset cleaning.
    """

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    ).reset_index(drop=True)

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Numerical columns
    numerical_columns = [
        "Traffic_Density",
        "Speed_Limit",
        "Number_of_Vehicles",
        "Driver_Alcohol",
        "Driver_Age",
        "Driver_Experience",
        "Accident"
    ]

    # Convert numerical columns
    for col in numerical_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features_target(df):
    """
    Separate features (X) and target (y).

    Accident is the target variable.

    Accident_Severity is removed because it can
    cause target leakage.
    """

    df = df.copy()

    # Remove rows where target is missing
    df = df.dropna(
        subset=["Accident"]
    ).reset_index(drop=True)

    # Convert target to integer
    df["Accident"] = df["Accident"].astype(int)

    # Separate target
    X = df.drop(
        columns=["Accident"]
    )

    y = df["Accident"]

    # Remove target leakage
    if "Accident_Severity" in X.columns:

        X = X.drop(
            columns=["Accident_Severity"]
        )

    return X, y


# ============================================================
# 5. GET FEATURE TYPES
# ============================================================

def get_feature_types(X):
    """
    Identify numerical and categorical features.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return (
        numerical_features,
        categorical_features
    )


# ============================================================
# 6. CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Create preprocessing pipeline.

    Numerical:
        Missing values → median
        Scaling → StandardScaler

    Categorical:
        Missing values → most frequent
        Encoding → OneHotEncoder
    """

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor
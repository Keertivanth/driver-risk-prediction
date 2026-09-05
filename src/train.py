"""Model training utilities."""
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.preprocessing import (
    load_data,
    clean_column_names,
    clean_data,
    prepare_features_target,
    get_feature_types,
    create_preprocessor
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "driver_risk.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_driver_risk_model.pkl"
)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("=" * 60)
    print("DRIVER RISK PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1] Loading dataset...")

    df = load_data(DATA_PATH)

    print(
        "Dataset shape:",
        df.shape
    )


    # --------------------------------------------------------
    # 2. CLEAN COLUMN NAMES
    # --------------------------------------------------------

    print("\n[2] Cleaning column names...")

    df = clean_column_names(df)


    # --------------------------------------------------------
    # 3. CLEAN DATA
    # --------------------------------------------------------

    print("\n[3] Cleaning dataset...")

    df = clean_data(df)

    print(
        "Cleaned dataset shape:",
        df.shape
    )


    # --------------------------------------------------------
    # 4. PREPARE FEATURES AND TARGET
    # --------------------------------------------------------

    print("\n[4] Preparing features and target...")

    X, y = prepare_features_target(df)

    print(
        "Features:",
        X.shape
    )

    print(
        "Target:",
        y.shape
    )


    # --------------------------------------------------------
    # 5. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\n[5] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # --------------------------------------------------------
    # 6. IDENTIFY FEATURE TYPES
    # --------------------------------------------------------

    print("\n[6] Identifying feature types...")

    numerical_features, categorical_features = (
        get_feature_types(X_train)
    )

    print(
        "Numerical features:",
        numerical_features
    )

    print(
        "Categorical features:",
        categorical_features
    )


    # --------------------------------------------------------
    # 7. CREATE PREPROCESSOR
    # --------------------------------------------------------

    print("\n[7] Creating preprocessing pipeline...")

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )


    # --------------------------------------------------------
    # 8. CREATE FINAL MODEL
    # --------------------------------------------------------

    print("\n[8] Creating final model...")

    # IMPORTANT:
    # Replace these parameters with the best parameters
    # obtained from 04_hyperparameter_tuning.ipynb.

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )


    # --------------------------------------------------------
    # 9. CREATE COMPLETE PIPELINE
    # --------------------------------------------------------

    print("\n[9] Creating complete ML pipeline...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


    # --------------------------------------------------------
    # 10. TRAIN
    # --------------------------------------------------------

    print("\n[10] Training model...")

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Model training completed!"
    )


    # --------------------------------------------------------
    # 11. SAVE MODEL
    # --------------------------------------------------------

    print("\n[11] Saving model...")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(
        "Model saved successfully!"
    )

    print(
        "Location:",
        MODEL_PATH
    )


    # --------------------------------------------------------
    # 12. RETURN OBJECTS
    # --------------------------------------------------------

    return (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()
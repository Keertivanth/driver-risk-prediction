"""Model evaluation utilities."""
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_driver_risk_model.pkl"
)

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test_data.csv"
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained ML pipeline.
    """

    model = joblib.load(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Load the test dataset.
    """

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )

    return test_df


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, test_df):
    """
    Evaluate the trained model
    on the untouched test dataset.
    """

    # Separate features and target
    X_test = test_df.drop(
        columns=["Accident"]
    )

    y_test = test_df["Accident"]

    # Remove target leakage
    if "Accident_Severity" in X_test.columns:

        X_test = X_test.drop(
            columns=["Accident_Severity"]
        )

    # Predictions
    y_pred = model.predict(
        X_test
    )

    # Probability of Accident
    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # Store metrics
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

    return metrics, y_test, y_pred


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):
    """
    Save evaluation metrics to CSV.
    """

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Score": list(metrics.values())
        }
    )

    output_path = os.path.join(
        REPORTS_DIR,
        "final_model_metrics.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DRIVER RISK MODEL EVALUATION")
    print("=" * 60)

    # Load model
    model = load_model()

    print("\nModel loaded successfully.")

    # Load test data
    test_df = load_test_data()

    print(
        "Test data loaded:",
        test_df.shape
    )

    # Evaluate
    metrics, y_test, y_pred = evaluate_model(
        model,
        test_df
    )

    # Display metrics
    print("\nEvaluation Results")
    print("-" * 40)

    for metric, score in metrics.items():

        print(
            f"{metric}: {score:.4f}"
        )

    # Confusion matrix
    print("\nConfusion Matrix")
    print("-" * 40)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # Classification report
    print("\nClassification Report")
    print("-" * 40)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Accident",
                "Accident"
            ],
            zero_division=0
        )
    )

    # Save metrics
    output_path = save_metrics(
        metrics
    )

    print(
        "\nMetrics saved to:"
    )

    print(output_path)
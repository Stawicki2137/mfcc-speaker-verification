from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import load_dataset


def create_models() -> dict:
    """
    Creates machine learning models used for comparison.

    Returns:
        Dictionary where key is model name and value is sklearn Pipeline/model.
    """
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "k-NN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluates trained model on test data.

    Args:
        model: Trained sklearn model.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["other", "target"],
            zero_division=0,
        ),
    }


def save_metrics_report(
    results: dict,
    output_path: Path,
) -> None:
    """
    Saves model comparison results to a text file.

    Args:
        results: Dictionary with model results.
        output_path: Path to output metrics file.
    """
    lines = []

    lines.append("Speaker Verification - Model Comparison")
    lines.append("=" * 45)
    lines.append("")

    for model_name, metrics in results.items():
        lines.append(f"Model: {model_name}")
        lines.append("-" * 45)
        lines.append(f"Accuracy : {metrics['accuracy']:.4f}")
        lines.append(f"Precision: {metrics['precision']:.4f}")
        lines.append(f"Recall   : {metrics['recall']:.4f}")
        lines.append(f"F1-score : {metrics['f1']:.4f}")
        lines.append("")
        lines.append("Confusion matrix:")
        lines.append(str(metrics["confusion_matrix"]))
        lines.append("")
        lines.append("Classification report:")
        lines.append(metrics["classification_report"])
        lines.append("")
        lines.append("=" * 45)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    target_dir = project_root / "data" / "raw" / "target"
    others_dir = project_root / "data" / "raw" / "others"

    models_dir = project_root / "models"
    results_dir = project_root / "results"

    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print("Loading dataset...")

    X, y, file_paths = load_dataset(
        target_dir=target_dir,
        others_dir=others_dir,
    )

    print("Dataset loaded.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target samples:", int(np.sum(y == 1)))
    print("Other samples:", int(np.sum(y == 0)))

    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain both classes: target and others.")

    if X.shape[0] < 4:
        raise ValueError("Dataset is too small. Add more recordings before training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    models = create_models()

    results = {}
    trained_models = {}

    print("\nTraining models...")

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        results[model_name] = metrics
        trained_models[model_name] = model

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1']:.4f}")
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

    best_model_name = max(results, key=lambda name: results[name]["f1"])
    best_model = trained_models[best_model_name]

    best_model_path = models_dir / "best_model.pkl"
    svm_model_path = models_dir / "svm_model.pkl"
    metrics_path = results_dir / "metrics.txt"

    joblib.dump(best_model, best_model_path)

    if "SVM" in trained_models:
        joblib.dump(trained_models["SVM"], svm_model_path)

    save_metrics_report(
        results=results,
        output_path=metrics_path,
    )

    print("\nTraining finished.")
    print("Best model:", best_model_name)
    print("Best model saved to:", best_model_path)
    print("SVM model saved to:", svm_model_path)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
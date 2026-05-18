from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from dataset import load_dataset


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    class_names: list[str],
) -> None:
    """
    Saves confusion matrix as an image.

    Args:
        cm: Confusion matrix.
        output_path: Path where the image should be saved.
        class_names: Names of classes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_evaluation_report(
    output_path: Path,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    cm: np.ndarray,
    report: str,
    false_accept: int,
    false_reject: int,
) -> None:
    """
    Saves evaluation metrics to a text file.
    """
    lines = []

    lines.append("Speaker Verification - Evaluation")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Accuracy : {accuracy:.4f}")
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall   : {recall:.4f}")
    lines.append(f"F1-score : {f1:.4f}")
    lines.append("")
    lines.append("Confusion matrix:")
    lines.append(str(cm))
    lines.append("")
    lines.append("Confusion matrix interpretation:")
    lines.append(f"True Negative  (other correctly rejected): {cm[0, 0]}")
    lines.append(f"False Accept   (other accepted as target): {false_accept}")
    lines.append(f"False Reject   (target rejected as other): {false_reject}")
    lines.append(f"True Positive  (target correctly accepted): {cm[1, 1]}")
    lines.append("")
    lines.append("Classification report:")
    lines.append(report)

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    target_dir = project_root / "data" / "raw" / "target"
    others_dir = project_root / "data" / "raw" / "others"

    model_path = project_root / "models" / "best_model.pkl"

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    evaluation_report_path = results_dir / "evaluation.txt"
    confusion_matrix_path = results_dir / "confusion_matrix.png"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train.py first."
        )

    print("Loading dataset...")

    X, y, file_paths = load_dataset(
        target_dir=target_dir,
        others_dir=others_dir,
    )

    print("Dataset loaded.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Loading model...")
    model = joblib.load(model_path)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    false_accept = int(cm[0, 1])
    false_reject = int(cm[1, 0])

    report = classification_report(
        y_test,
        y_pred,
        target_names=["other", "target"],
        zero_division=0,
    )

    print("\nEvaluation results")
    print("------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("")
    print(f"False Accept: {false_accept}")
    print(f"False Reject: {false_reject}")

    save_evaluation_report(
        output_path=evaluation_report_path,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cm=cm,
        report=report,
        false_accept=false_accept,
        false_reject=false_reject,
    )

    plot_confusion_matrix(
        cm=cm,
        output_path=confusion_matrix_path,
        class_names=["other", "target"],
    )

    print("\nEvaluation report saved to:", evaluation_report_path)
    print("Confusion matrix image saved to:", confusion_matrix_path)


if __name__ == "__main__":
    main()
from pathlib import Path
import argparse

import joblib
import numpy as np

from features import extract_feature_vector_from_file


TARGET_LABEL = 1
OTHER_LABEL = 0


def get_target_probability(model, feature_vector: np.ndarray) -> float:
    """
    Returns probability that the recording belongs to the target speaker.

    Args:
        model: Trained sklearn model.
        feature_vector: Feature vector with shape (1, num_features).

    Returns:
        Probability of target speaker class.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)

        class_labels = list(model.classes_)

        if TARGET_LABEL not in class_labels:
            raise ValueError("Target label not found in model classes.")

        target_index = class_labels.index(TARGET_LABEL)

        return float(probabilities[0][target_index])

    prediction = model.predict(feature_vector)[0]

    return 1.0 if prediction == TARGET_LABEL else 0.0


def predict_speaker(
    wav_path: str | Path,
    model_path: str | Path,
    threshold: float = 0.5
) -> tuple[int, float]:
    """
    Predicts whether a WAV file belongs to the target speaker.

    Args:
        wav_path: Path to WAV file.
        model_path: Path to trained model.
        threshold: Decision threshold for target speaker.

    Returns:
        prediction: 1 for target speaker, 0 for other speaker.
        target_probability: Probability of target speaker.
    """
    wav_path = Path(wav_path)
    model_path = Path(model_path)

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    _, feature_vector = extract_feature_vector_from_file(wav_path)

    feature_vector = feature_vector.reshape(1, -1)

    target_probability = get_target_probability(model, feature_vector)

    prediction = TARGET_LABEL if target_probability >= threshold else OTHER_LABEL

    return prediction, target_probability


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Predict whether a WAV recording belongs to the target speaker."
    )

    parser.add_argument(
        "wav_path",
        type=str,
        help="Path to WAV file."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(project_root / "models" / "best_model.pkl"),
        help="Path to trained model. Default: models/best_model.pkl"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for target speaker. Default: 0.5"
    )

    args = parser.parse_args()

    prediction, probability = predict_speaker(
        wav_path=args.wav_path,
        model_path=args.model,
        threshold=args.threshold
    )

    label_name = "TARGET SPEAKER" if prediction == TARGET_LABEL else "OTHER SPEAKER"

    print("Prediction:", label_name)
    print(f"Target probability: {probability:.4f}")
    print(f"Decision threshold: {args.threshold:.2f}")

    if prediction == TARGET_LABEL:
        print("Decision: YES - this is the target speaker.")
    else:
        print("Decision: NO - this is not the target speaker.")


if __name__ == "__main__":
    main()
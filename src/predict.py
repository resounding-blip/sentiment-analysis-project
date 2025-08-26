import argparse
from typing import Any, Iterable
import numpy as np
from numpy.typing import NDArray
from joblib import load

def main(model_path: str, input_texts: list[str]) -> None:
    """Loads a model to predict and show probabilities for a list of texts."""

    classifier: Any = load(model_path)

    predictions: NDArray[Any] = classifier.predict(input_texts)
    
    probabilities: NDArray[np.float64] | None = None
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(input_texts)[:, 1]

    probabilities_list: list[np.float64 | None]
    probabilities_list = probabilities if probabilities is not None else [None] * len(input_texts)

    for text, prediction, probability in zip(input_texts, predictions, probabilities_list):
        if probability is None:
            print(f"{prediction}\t{text}")
        else:
            print(f"{prediction}\t{probability:.3f}\t{text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    
    args: argparse.Namespace = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)
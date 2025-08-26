import argparse
from typing import Any, Iterable
import numpy as np
from numpy.typing import NDArray
from joblib import load

def main(model_path: str, input_texts: list[str]) -> None:
    """Loads a model to predict and show probabilities for a list of texts."""
    # Logic will go here later
    print("Prediction script is running...")
    print(f"Model path: {model_path}")
    print(f"Texts: {input_texts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    
    args: argparse.Namespace = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)
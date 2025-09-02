# tests/test_classifier_basic.py

from pathlib import Path
from joblib import load
import pytest

MODEL_PATH = Path("models/sentiment.joblib")

def test_good_is_positive():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}. Train it first.")
    classifier = load(MODEL_PATH)
    assert classifier.predict(["good"])[0] == 1

def test_bad_is_negative():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}. Train it first.")
    classifier = load(MODEL_PATH)
    assert classifier.predict(["bad"])[0] == 0

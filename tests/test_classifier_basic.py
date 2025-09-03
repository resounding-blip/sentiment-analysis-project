from src.predict import load_model, predict_texts


MODEL_PATH = "models/sentiment.joblib"


def test_positive_sentence_is_positive() -> None:
    clf = load_model(MODEL_PATH)
    text = "I love this movie, it was fantastic and inspiring!"
    preds, _ = predict_texts(clf, [text])
    assert preds[0] == 1


def test_negative_sentence_is_negative() -> None:
    clf = load_model(MODEL_PATH)
    text = "The service was terrible and the food was awful."
    preds, _ = predict_texts(clf, [text])
    assert preds[0] == 0

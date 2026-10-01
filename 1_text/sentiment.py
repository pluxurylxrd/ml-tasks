import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"

_tokenizer = None
_model = None


def load_model():
    """Грузит модель один раз и дальше отдаёт уже загруженную."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def predict(text: str) -> dict:
    """Возвращает тональность отзыва и вероятности по всем трём классам."""
    if not text or not text.strip():
        raise ValueError("Текст не может быть пустым")

    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    proba = torch.softmax(logits, dim=1)[0]
    best = int(proba.argmax())

    return {
        "label": model.config.id2label[best],
        "score": round(float(proba[best]), 4),
        "probabilities": {
            model.config.id2label[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        },
    }


if __name__ == "__main__":
    examples = [
        "Заказ пришёл за два дня, упаковано отлично, продавцу спасибо",
        "Товар бракованный, продавец на сообщения не отвечает, деньги не вернули",
        "Размер 42, цвет чёрный, доставка до пункта выдачи",
    ]
    for text in examples:
        result = predict(text)
        print(f"{text}\n  → {result['label']} ({result['score']})\n")
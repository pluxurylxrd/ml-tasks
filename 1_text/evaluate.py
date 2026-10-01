import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentiment import predict, load_model

DATA_PATH = Path(__file__).parent / "data" / "test_samples.json"


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    start = time.perf_counter()
    load_model()
    load_time = time.perf_counter() - start

    y_true, y_pred = [], []

    start = time.perf_counter()
    for sample in samples:
        y_true.append(sample["label"])
        y_pred.append(predict(sample["text"])["label"])
    predict_time = time.perf_counter() - start

    print(f"Загрузка модели: {load_time:.2f} с")
    print(f"Предсказание {len(samples)} отзывов: {predict_time:.2f} с")
    print(f"В среднем на отзыв: {predict_time / len(samples) * 1000:.1f} мс\n")

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    labels = ["negative", "neutral", "positive"]
    print("Матрица ошибок (строки - правда, столбцы - предсказание):")
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in labels))
    for label, row in zip(labels, confusion_matrix(y_true, y_pred, labels=labels)):
        print(f"{label:>10}" + "".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    main()
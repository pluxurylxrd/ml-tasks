import time
from pathlib import Path

from classify import predict, load_model

DATA_DIR = Path(__file__).parent / "data"

FOLDER_TO_LABEL = {
    "backpack": "backpack",
    "coffee_mug": "coffee mug",
    "running_shoe": "running shoe",
    "digital_watch": "digital watch",
    "umbrella": "umbrella",
    "teapot": "teapot",
    "wallet": "wallet",
    "water_bottle": "water bottle",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images():
    """Собирает пары (путь к файлу, правильный класс) из папок data/."""
    items = []
    for folder, label in FOLDER_TO_LABEL.items():
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            print(f"Пропущена отсутствующая папка: {folder}")
            continue
        for file in sorted(folder_path.iterdir()):
            if file.suffix.lower() in EXTENSIONS:
                items.append((file, label))
    return items


def main():
    items = collect_images()
    if not items:
        print("Изображения не найдены. Проверь папку data/")
        return

    start = time.perf_counter()
    load_model()
    load_time = time.perf_counter() - start

    top1_hits = 0
    top5_hits = 0
    errors = []

    start = time.perf_counter()
    for file, true_label in items:
        result = predict(str(file), top_k=5)
        top5_labels = [p["label"] for p in result["top_k"]]

        if result["label"] == true_label:
            top1_hits += 1
        else:
            errors.append((file.parent.name + "/" + file.name, true_label, result["label"], result["score"]))

        if true_label in top5_labels:
            top5_hits += 1
    predict_time = time.perf_counter() - start

    total = len(items)

    print(f"\nЗагрузка модели: {load_time:.2f} с")
    print(f"Классификация {total} изображений: {predict_time:.2f} с")
    print(f"В среднем на изображение: {predict_time / total * 1000:.1f} мс\n")

    print(f"Top-1 accuracy: {top1_hits / total:.3f}  ({top1_hits} из {total})")
    print(f"Top-5 accuracy: {top5_hits / total:.3f}  ({top5_hits} из {total})\n")

    if errors:
        print("Ошибки top-1:")
        for name, true_label, got, score in errors:
            print(f"  {name:<30} ожидалось: {true_label:<15} получено: {got} ({score})")


if __name__ == "__main__":
    main()
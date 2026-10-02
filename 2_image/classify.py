from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

_model = None
_preprocess = None
_categories = None


def load_model():
    """Грузит ResNet-50 с весами ImageNet один раз."""
    global _model, _preprocess, _categories
    if _model is None:
        weights = ResNet50_Weights.IMAGENET1K_V2
        _model = resnet50(weights=weights)
        _model.eval()
        _preprocess = weights.transforms()
        _categories = weights.meta["categories"]
    return _model, _preprocess, _categories


def predict(image_path: str, top_k: int = 5) -> dict:
    """Классифицирует изображение товара и возвращает top_k вариантов."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    model, preprocess, categories = load_model()

    image = Image.open(path).convert("RGB")
    batch = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(batch)

    proba = torch.softmax(logits, dim=1)[0]
    scores, indices = proba.topk(top_k)

    predictions = [
        {"label": categories[idx], "score": round(float(score), 4)}
        for score, idx in zip(scores, indices)
    ]

    return {
        "file": path.name,
        "label": predictions[0]["label"],
        "score": predictions[0]["score"],
        "top_k": predictions,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python classify.py путь_к_картинке")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"\n{result['file']}")
    print(f"Ответ: {result['label']} ({result['score']})\n")
    print("Топ-5:")
    for i, p in enumerate(result["top_k"], 1):
        print(f"  {i}. {p['label']:<25} {p['score']}")
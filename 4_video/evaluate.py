import json
import time
from collections import Counter
from pathlib import Path

from detect import detect_video, load_model

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "results.json"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
IGNORED_LABELS = {"person"}


def analyze(video_result: dict, expected: str) -> dict:
    """Считает метрики по одному видео"""
    frames = video_result["frames"]
    total = len(frames)

    frames_with_any = 0
    frames_with_expected = 0
    expected_scores = []
    top_labels = []
    all_labels = Counter()

    for item in frames:
        detections = [d for d in item["detections"] if d["label"] not in IGNORED_LABELS]

        if detections:
            frames_with_any += 1
            best = max(detections, key=lambda d: d["score"])
            top_labels.append(best["label"])
        else:
            top_labels.append(None)

        for d in detections:
            all_labels[d["label"]] += 1

        matched = [d for d in detections if d["label"] == expected]
        if matched:
            frames_with_expected += 1
            expected_scores.append(max(d["score"] for d in matched))

    switches = sum(
        1 for a, b in zip(top_labels, top_labels[1:]) if a != b
    )

    return {
        "expected": expected,
        "processed_frames": total,
        "detection_rate": round(frames_with_any / total, 3) if total else 0.0,
        "correct_rate": round(frames_with_expected / total, 3) if total else 0.0,
        "mean_confidence": round(sum(expected_scores) / len(expected_scores), 4)
        if expected_scores else 0.0,
        "label_switches": switches,
        "stability": round(1 - switches / (total - 1), 3) if total > 1 else 1.0,
        "all_labels": dict(all_labels.most_common()),
    }


def main():
    videos = sorted(
        f for f in DATA_DIR.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print("Видео не найдены. Положи файлы в data/")
        return

    start = time.perf_counter()
    load_model()
    load_time = time.perf_counter() - start

    results = {}
    total_processed = 0

    start = time.perf_counter()
    for video in videos:
        expected = video.stem.replace("_", " ")
        detection = detect_video(str(video))
        stats = analyze(detection, expected)

        stats["fps"] = detection["fps"]
        stats["total_frames"] = detection["total_frames"]
        results[video.name] = stats
        total_processed += stats["processed_frames"]

        print(f"\n{video.name}  (ожидается: {expected})")
        print(f"  обработано кадров:        {stats['processed_frames']} из {stats['total_frames']}")
        print(f"  кадров с детекцией:       {stats['detection_rate']}")
        print(f"  кадров с верным классом:  {stats['correct_rate']}")
        print(f"  средняя уверенность:      {stats['mean_confidence']}")
        print(f"  смен основного класса:    {stats['label_switches']}")
        print(f"  стабильность:             {stats['stability']}")
        print(f"  все найденные классы:     {stats['all_labels']}")
    predict_time = time.perf_counter() - start

    print(f"\n{'-' * 60}")
    print(f"Загрузка модели: {load_time:.2f} с")
    print(f"Обработка {len(videos)} видео, {total_processed} кадров: {predict_time:.2f} с")
    print(f"В среднем на кадр: {predict_time / total_processed * 1000:.1f} мс")

    mean_correct = sum(r["correct_rate"] for r in results.values()) / len(results)
    mean_stability = sum(r["stability"] for r in results.values()) / len(results)
    print(f"\nСредняя доля верных кадров: {mean_correct:.3f}")
    print(f"Средняя стабильность:       {mean_stability:.3f}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {OUT_PATH.name}")


if __name__ == "__main__":
    main()
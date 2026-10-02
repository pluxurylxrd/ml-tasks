from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.35

_model = None


def load_model():
    """Грузит YOLO один раз"""
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


def detect_video(video_path: str, step: int = 5, save_annotated: bool = True) -> dict:
    """Прогоняет видео по кадрам и возвращает детекции на каждом обработанном кадре"""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    model = load_model()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_annotated:
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{path.stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps / step, (width, height))

    frames = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % step != 0:
            frame_index += 1
            continue

        result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        detections = []
        for box in result.boxes:
            detections.append({
                "label": result.names[int(box.cls)],
                "score": round(float(box.conf), 4),
            })

        frames.append({"frame": frame_index, "detections": detections})

        if writer is not None:
            writer.write(result.plot())

        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()

    return {
        "file": path.name,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "processed_frames": len(frames),
        "step": step,
        "frames": frames,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python detect.py путь_к_видео")
        sys.exit(1)

    result = detect_video(sys.argv[1])
    print(f"\n{result['file']}")
    print(f"  {result['fps']} кадров/с, всего кадров {result['total_frames']}")
    print(f"  обработано каждый {result['step']}-й: {result['processed_frames']}\n")

    for item in result["frames"][:10]:
        labels = ", ".join(f"{d['label']} {d['score']}" for d in item["detections"])
        print(f"  кадр {item['frame']:>4}: {labels or '—'}")
import json
from pathlib import Path

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DATA_DIR = Path(__file__).parent / "data"

TEXTS = [
    "Заказ пришёл за три дня, упаковка целая, продавцу большое спасибо.",
    "Куртка села после первой стирки, хотя стирал строго по бирке.",
    "Размер сорок два, цвет чёрный, состав восемьдесят процентов хлопок.",
    "Брала на пробу, в итоге заказала ещё две штуки родителям.",
    "Товар пришёл с трещиной на корпусе, продавец на сообщения не отвечает.",
    "Пункт выдачи работает с девяти до двадцати одного, вход со двора.",
    "Качество отличное, цена заметно ниже, чем в магазине у дома.",
    "В комплекте шнур полтора метра, адаптер и инструкция, батарейки отдельно.",
]


def record_one(path: Path):
    """Пишет с микрофона, пока пользователь не нажмёт Enter."""
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status)
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        input("  Идёт запись... Enter — стоп ")

    if not frames:
        print("  Ничего не записалось, повтори")
        return False

    import numpy as np
    audio = np.concatenate(frames, axis=0)
    sf.write(path, audio, SAMPLE_RATE)

    duration = len(audio) / SAMPLE_RATE
    print(f"  Сохранено: {path.name} ({duration:.1f} с)\n")
    return True


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    reference = {}

    for i, text in enumerate(TEXTS, 1):
        path = DATA_DIR / f"{i:02d}.wav"

        while True:
            print(f"\n[{i}/{len(TEXTS)}] Прочитай вслух:")
            print(f"  {text}\n")
            input("  Enter — начать запись ")

            if not record_one(path):
                continue

            answer = input("  Enter — дальше, п — перезаписать: ").strip().lower()
            if answer != "п":
                break

        reference[path.name] = text

    ref_path = DATA_DIR / "reference.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Записано файлов: {len(reference)}")
    print(f"Эталонные тексты: {ref_path}")


if __name__ == "__main__":
    main()
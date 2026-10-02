import json
import re
import time
from pathlib import Path

import jiwer
from num2words import num2words

from transcribe import transcribe, load_model

DATA_DIR = Path(__file__).parent / "data"
REF_PATH = DATA_DIR / "reference.json"
OUT_PATH = Path(__file__).parent / "results.json"


def normalize(text: str) -> str:
    """Приводит текст к виду, в котором сравнение осмысленно."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = []
    for token in text.split():
        digits = re.match(r"^(\d+)", token)
        if digits:
            tokens.append(num2words(int(digits.group(1)), lang="ru"))
        else:
            tokens.append(token)

    return " ".join(" ".join(tokens).split())


def main():
    with open(REF_PATH, encoding="utf-8") as f:
        reference = json.load(f)

    start = time.perf_counter()
    load_model()
    load_time = time.perf_counter() - start

    refs_raw, hyps_raw = [], []
    refs_norm, hyps_norm = [], []
    total_audio = 0.0
    results = {}

    start = time.perf_counter()
    for filename in sorted(reference):
        true_text = reference[filename]
        result = transcribe(str(DATA_DIR / filename))
        hyp_text = result["text"]
        total_audio += result["duration"]

        refs_raw.append(true_text)
        hyps_raw.append(hyp_text)
        refs_norm.append(normalize(true_text))
        hyps_norm.append(normalize(hyp_text))

        file_wer = jiwer.wer(normalize(true_text), normalize(hyp_text))
        results[filename] = {
            "reference": true_text,
            "hypothesis": hyp_text,
            "wer": round(file_wer, 3),
        }

        print(f"\n{filename}  WER {file_wer:.3f}")
        print(f"  эталон:   {true_text}")
        print(f"  распознано: {hyp_text}")
    predict_time = time.perf_counter() - start

    wer_raw = jiwer.wer(refs_raw, hyps_raw)
    wer_norm = jiwer.wer(refs_norm, hyps_norm)
    cer_norm = jiwer.cer(refs_norm, hyps_norm)
    stats = jiwer.process_words(refs_norm, hyps_norm)

    print(f"\n{'-' * 60}")
    print(f"Загрузка модели: {load_time:.2f} с")
    print(f"Распознавание {len(reference)} файлов: {predict_time:.2f} с")
    print(f"Суммарная длительность аудио: {total_audio:.1f} с")
    print(f"Real-time factor: {predict_time / total_audio:.2f}\n")

    print(f"WER без нормализации: {wer_raw:.3f}")
    print(f"WER с нормализацией:  {wer_norm:.3f}")
    print(f"CER с нормализацией:  {cer_norm:.3f}\n")

    print("Типы ошибок (после нормализации):")
    print(f"  замены:   {stats.substitutions}")
    print(f"  пропуски: {stats.deletions}")
    print(f"  вставки:  {stats.insertions}")
    print(f"  верных слов: {stats.hits}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРасшифровки сохранены: {OUT_PATH.name}")


if __name__ == "__main__":
    main()
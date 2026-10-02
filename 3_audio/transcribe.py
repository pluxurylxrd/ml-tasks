from pathlib import Path

import soundfile as sf
from transformers import pipeline

MODEL_NAME = "openai/whisper-small"
SAMPLE_RATE = 16000

_pipe = None


def load_model():
    """Грузит Whisper один раз."""
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=-1,
        )
    return _pipe


def transcribe(audio_path: str) -> dict:
    """Переводит речь из WAV-файла в текст."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    audio, sample_rate = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Ожидается частота {SAMPLE_RATE} Гц, в файле {sample_rate} Гц"
        )

    pipe = load_model()
    result = pipe(
        {"raw": audio, "sampling_rate": sample_rate},
        generate_kwargs={"language": "russian", "task": "transcribe"},
    )

    return {
        "file": path.name,
        "text": result["text"].strip(),
        "duration": round(len(audio) / sample_rate, 2),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python transcribe.py путь_к_wav")
        sys.exit(1)

    result = transcribe(sys.argv[1])
    print(f"\n{result['file']} ({result['duration']} с)")
    print(f"  {result['text']}\n")
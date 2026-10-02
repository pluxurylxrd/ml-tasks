from pathlib import Path

import numpy as np
import soundfile as sf

DATA_DIR = Path(__file__).parent / "data"

for path in sorted(DATA_DIR.glob("*.wav")):
    audio, sr = sf.read(path, dtype="float32")
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio ** 2).mean()))
    print(f"{path.name}  пик {peak:.3f}  средняя громкость {rms:.4f}  {len(audio)/sr:.1f} с")
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import soundfile as sf

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_wav_int16(path: Path, y: np.ndarray, sr: int):
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=0)
    y16 = np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)
    sf.write(str(path), y16, sr, subtype='PCM_16', format='WAV')

def save_jsonl(path: Path, items):
    with path.open('w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

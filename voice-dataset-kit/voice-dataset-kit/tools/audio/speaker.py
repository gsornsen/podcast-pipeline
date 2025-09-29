from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa
try:
    import torch
    from speechbrain.pretrained import EncoderClassifier
except Exception:
    torch = None

class SpeakerFilter:
    def __init__(self, ref_dir: Path|None, sr=24000, thresh=0.5):
        self.enabled=False; self.sr=sr; self.thresh=float(thresh)
        self.ref_emb=None; self.cls=None
        if ref_dir and ref_dir.exists() and torch is not None:
            try:
                self.cls = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"}
                )
                embs=[]
                for wav in sorted(ref_dir.glob('*.wav')):
                    y,_ = librosa.load(str(wav), sr=self.sr, mono=True)
                    t = torch.tensor(y).unsqueeze(0)
                    with torch.no_grad():
                        e = self.cls.encode_batch(t).squeeze(0).squeeze(0).cpu().numpy()
                    e = e/(np.linalg.norm(e)+1e-9)
                    embs.append(e)
                if embs:
                    m = np.mean(np.stack(embs), axis=0)
                    self.ref_emb = m/(np.linalg.norm(m)+1e-9)
                    self.enabled=True
            except Exception as e:
                print(f"[speaker] disabled: {e}")
    def score(self, y: np.ndarray) -> float:
        if not self.enabled: return 1.0
        t = torch.tensor(y).unsqueeze(0)
        with torch.no_grad():
            e = self.cls.encode_batch(t).squeeze(0).squeeze(0).cpu().numpy()
        e = e/(np.linalg.norm(e)+1e-9)
        return float(np.dot(self.ref_emb, e))
    def accept(self, y: np.ndarray) -> bool:
        return self.score(y) >= self.thresh

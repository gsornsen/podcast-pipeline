from __future__ import annotations
from pathlib import Path
import librosa
import numpy as np
try:
    import torch
except Exception:
    torch = None
from .qc import trim_with_margin, rms_db, snr_db, has_clipping, detect_hum, detect_plosives, auto_notch
from ..subtitles.parse import load_for_audio, windows_from_subs

def load_vad():
    if torch is None:
        raise RuntimeError('PyTorch required for Silero VAD')
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, *_) = utils
    return model, get_speech_timestamps

def vad_segments(y, sr, min_speech_ms=200, min_silence_ms=150, max_pause_ms=400):
    model, get_speech_timestamps = load_vad()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    wav = torch.from_numpy(y).float().to(dev)
    ts = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=False,
                               threshold=0.5, min_speech_duration_ms=min_speech_ms,
                               min_silence_duration_ms=min_silence_ms)
    merged = []
    for seg in ts:
        s, e = int(seg['start']), int(seg['end'])
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s - pe <= int(max_pause_ms*sr/1000):
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
    return merged

class Clip:
    def __init__(self, doc_id, start_s, end_s, text, qc):
        self.doc_id=doc_id; self.start_s=start_s; self.end_s=end_s
        self.text=text; self.qc=qc

def segment_file(audio_path: Path, sr=24000, min_sec=5.0, max_sec=15.0, margin_ms=150,
                 mains_hz: int|None=None, apply_notch=False, notch_strength=0.7,
                 speaker_filter=None):
    y, _sr = librosa.load(str(audio_path), sr=sr, mono=True)
    if apply_notch:
        y = auto_notch(y, sr, mains_hz=mains_hz, strength=notch_strength)

    subs = load_for_audio(audio_path)
    clips = []

    def try_add(st, et, txt):
        seg = y[int(st*sr):int(et*sr)]
        seg = trim_with_margin(seg, sr, margin_ms)
        dur = len(seg)/sr
        if dur < min_sec: return
        qc = {"rms_db": rms_db(seg), "snr_db": snr_db(seg), "clipped": 1.0 if has_clipping(seg) else 0.0}
        qc.update(detect_hum(seg, sr, mains_hz))
        qc.update(detect_plosives(seg, sr))
        if qc["rms_db"] < -40 or qc["snr_db"] < 5.0 or qc["clipped"] > 0.0:
            return
        if speaker_filter is not None and getattr(speaker_filter, 'enabled', False):
            if not speaker_filter.accept(seg): return
        clips.append(Clip(audio_path.stem, st, et, txt, qc))

    if subs:
        for st, et, txt in windows_from_subs(subs, min_sec, max_sec):
            try_add(st, et, txt)
    if not subs or not clips:
        # Fallback to VAD
        try:
            ts = vad_segments(y, sr)
        except Exception as e:
            print(f"[vad] failed on {audio_path.name}: {e}")
            ts = [(0, len(y))]
        for s0, s1 in ts:
            st = s0/sr
            while st + min_sec <= s1/sr:
                et = min(st + max_sec, s1/sr)
                try_add(st, et, None)
                st = et
    return clips, y

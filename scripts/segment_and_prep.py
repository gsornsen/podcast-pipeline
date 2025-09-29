#!/usr/bin/env python3
import argparse
import math
import os
import pathlib
import random
import subprocess
import sys
from pathlib import Path
from typing import list, tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import webrtcvad
from tqdm import tqdm

# ---------- Helpers ----------


def load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y


def write_wav_16bit(path: str, y: np.ndarray, sr: int):
    # 16-bit PCM
    y_clipped = np.clip(y, -1.0, 1.0)
    sf.write(path, y_clipped, sr, subtype="PCM_16")


def lufs_normalize(y: np.ndarray, sr: int, target_lufs: float = -20.0) -> np.ndarray:
    meter = pyln.Meter(sr)  # EBU R128
    loudness = meter.integrated_loudness(y)
    if np.isfinite(loudness):
        gain = target_lufs - loudness
        y = pyln.normalize.loudness(y, loudness, target_lufs)
    return y


def frame_generator(y: np.ndarray, sr: int, frame_ms: int = 30):
    frame_len = int(sr * frame_ms / 1000.0)
    for start in range(0, len(y), frame_len):
        end = min(start + frame_len, len(y))
        yield y[start:end]


def vad_segments(
    y: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    pad_ms: int = 150,
    min_speech_ms: int = 200,
):
    """Return list of (start_idx, end_idx) for voiced segments."""
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000.0)
    pad_frames = int(pad_ms / frame_ms)
    is_speech = []
    # Convert to 16-bit PCM bytes for VAD
    int16 = np.int16(np.clip(y, -1, 1) * 32767)
    raw = int16.tobytes()

    # WebRTC VAD expects bytes per frame
    # 16-bit mono => 2 bytes per sample
    bytes_per_frame = frame_len * 2
    num_frames = len(raw) // bytes_per_frame

    for i in range(num_frames):
        chunk = raw[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        is_speech.append(vad.is_speech(chunk, sr))

    # Smooth + pad
    voiced = []
    start = None
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = i
        elif not speech and start is not None:
            end = i
            voiced.append((start, end))
            start = None
    if start is not None:
        voiced.append((start, len(is_speech)))

    # Merge close segments with padding
    merged = []
    for s, e in voiced:
        s = max(0, s - pad_frames)
        e = min(len(is_speech), e + pad_frames)
        if not merged:
            merged.append([s, e])
        else:
            prev_s, prev_e = merged[-1]
            if s <= prev_e + 1:
                merged[-1][1] = max(prev_e, e)
            else:
                merged.append([s, e])

    # Convert to sample indices and filter by min length
    segs = []
    min_frames = max(1, int(min_speech_ms / frame_ms))
    for s, e in merged:
        if (e - s) >= min_frames:
            segs.append((s * frame_len, e * frame_len))
    return segs


def cap_and_split_segments(
    y: np.ndarray, segs: list[tuple[int, int]], sr: int, min_sec: float, max_sec: float
):
    """Ensure segments fall within [min_sec, max_sec] by splitting long ones."""
    out = []
    min_len = int(min_sec * sr)
    max_len = int(max_sec * sr)
    for s, e in segs:
        length = e - s
        if length <= 0:
            continue
        cur = s
        while (e - cur) > max_len:
            out.append((cur, cur + max_len))
            cur += max_len
        if (e - cur) >= min_len:
            out.append((cur, e))
    return out


def ensure_sr_mono_24k(path_in: str, path_out: str, target_sr: int = 24000):
    y = load_audio_mono(path_in, target_sr)
    write_wav_16bit(path_out, y, target_sr)


def pick_refs(
    y: np.ndarray, sr: int, segs: list[tuple[int, int]], outdir: Path, max_refs: int = 6
):
    # Take a few mid-length segments (1–3s) as anchors
    cand = []
    for s, e in segs:
        dur = (e - s) / sr
        if 1.0 <= dur <= 3.0:
            cand.append((s, e))
    random.shuffle(cand)
    for i, (s, e) in enumerate(cand[:max_refs]):
        ref = y[s:e]
        write_wav_16bit(outdir / f"anchor_{i + 1:02d}.wav", ref, sr)


def find_audio_files(paths: list[str]):
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    files = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for ext in exts:
                files += list(pth.rglob(f"*{ext}"))
        elif pth.is_file() and pth.suffix.lower() in exts:
            files.append(pth)
    return sorted(files)


# ---------- Optional ASR ----------


def transcribe_whisper(
    path: str, engine: str = "openai", model_name: str = "base"
) -> str:
    try:
        if engine == "openai":
            import whisper

            model = whisper.load_model(model_name)
            result = model.transcribe(path)
            return result.get("text", "").strip()
        elif engine == "faster":
            from faster_whisper import WhisperModel

            model = WhisperModel(model_name, compute_type="float16")
            segments, info = model.transcribe(path)
            text = " ".join(seg.text for seg in segments).strip()
            return text
    except Exception as e:
        print(f"[WARN] ASR failed for {path}: {e}")
    return ""


# ---------- Main ----------


def main():
    ap = argparse.ArgumentParser(
        description="Segment long recordings into 5–15s utterances with VAD, normalize to LUFS, resample to 24k, and build metadata."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files or directories containing long takes",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Output data directory (contains train/val/test/ref)",
    )
    ap.add_argument("--sr", type=int, default=24000, help="Target sample rate")
    ap.add_argument("--lufs", type=float, default=-20.0, help="Target LUFS loudness")
    ap.add_argument(
        "--min-sec", type=float, default=5.0, help="Minimum utterance length (seconds)"
    )
    ap.add_argument(
        "--max-sec", type=float, default=15.0, help="Maximum utterance length (seconds)"
    )
    ap.add_argument(
        "--vad",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness (0-3)",
    )
    ap.add_argument(
        "--pad-ms", type=int, default=150, help="Padding (ms) around VAD segments"
    )
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for splits")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument(
        "--max-refs", type=int, default=6, help="Max reference anchors to export"
    )
    ap.add_argument(
        "--whisper",
        type=str,
        default="",
        help="Optional OpenAI Whisper model name, e.g. 'base' (pip install openai-whisper)",
    )
    ap.add_argument(
        "--faster-whisper",
        dest="faster_whisper",
        type=str,
        default="",
        help="Optional faster-whisper model name, e.g. 'base'",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    train_dir = outdir / "train"
    val_dir = outdir / "val"
    test_dir = outdir / "test"
    ref_dir = outdir / "ref"
    for d in [train_dir, val_dir, test_dir, ref_dir]:
        d.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(args.inputs)
    if not files:
        print("No input audio found.", file=sys.stderr)
        sys.exit(1)

    # Process each source file, keep segments grouped by source for splitting
    source_segments = []  # list of (source_name, [ (wav_path, rel_path, text) ... ])
    total_refs = 0

    for fpath in tqdm(files, desc="Segmenting"):
        y = load_audio_mono(str(fpath), args.sr)
        y = lufs_normalize(y, args.sr, args.lufs)

        segs = vad_segments(
            y, args.sr, args.vad, frame_ms=30, pad_ms=args.pad_ms, min_speech_ms=200
        )
        segs = cap_and_split_segments(y, segs, args.sr, args.min_sec, args.max_sec)

        # Export segments into a temporary list before assigning to splits
        items = []
        base_name = fpath.stem
        for i, (s, e) in enumerate(segs):
            utt = y[s:e]
            out_name = f"{base_name}_{i:05d}.wav"
            tmp_out = out_name  # name only; split assignment later
            items.append((utt, out_name))

        # Select a few refs
        if args.max_refs > 0 and total_refs < args.max_refs and len(segs) > 0:
            pick_refs(
                y, args.sr, segs, ref_dir, max_refs=min(args.max_refs - total_refs, 6)
            )
            total_refs += min(args.max_refs - total_refs, 6)

        source_segments.append((fpath.stem, items))

    # Split by source file to reduce leakage
    sources = list(range(len(source_segments)))
    random.shuffle(sources)
    n = len(sources)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_idx = set(sources[:n_train])
    val_idx = set(sources[n_train : n_train + n_val])
    test_idx = set(sources[n_train + n_val :])

    # Prepare metadata writers
    def open_meta(d: Path):
        mpath = d / "metadata.csv"
        exists = mpath.exists()
        f = open(mpath, "a", encoding="utf-8")
        if not exists:
            f.write("# rel_path|text\n")
        return f

    m_train = open_meta(train_dir)
    m_val = open_meta(val_dir)
    m_test = open_meta(test_dir)

    # Determine ASR engine
    asr_engine = None
    asr_model = None
    if args.whisper:
        asr_engine = "openai"
        asr_model = args.whisper
    elif args.faster_whisper:
        asr_engine = "faster"
        asr_model = args.faster_whisper

    # Write out segments and metadata
    for idx, (src_name, items) in enumerate(tqdm(source_segments, desc="Writing")):
        if idx in train_idx:
            d = train_dir
            meta_f = m_train
        elif idx in val_idx:
            d = val_dir
            meta_f = m_val
        else:
            d = test_dir
            meta_f = m_test

        for utt, out_name in items:
            out_path = d / out_name
            write_wav_16bit(out_path, utt, args.sr)
            text = ""
            if asr_engine is not None:
                text = transcribe_whisper(
                    str(out_path), engine=asr_engine, model_name=asr_model
                )
            # Pipe-separated
            rel = out_path.name
            meta_f.write(f"{rel}|{text}\n")

    for f in [m_train, m_val, m_test]:
        f.close()

    print(
        "Done. Review and fix transcripts in train/val/test metadata.csv, and keep a few anchors in data/ref/."
    )
    print(
        "Tip: spell out numbers (e.g., 'twenty twenty-five'), and keep punctuation that guides prosody."
    )


if __name__ == "__main__":
    main()

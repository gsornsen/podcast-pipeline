from __future__ import annotations
# Import the patch module first to fix torchaudio deprecation warning
import voice_dataset_kit  # This applies the monkey-patch that fixes torchaudio warning
from pathlib import Path
import argparse, json
from tqdm import tqdm
from ..tools.audio.segment import segment_file
from ..tools.audio.speaker import SpeakerFilter
from ..tools.util.io import ensure_dir, write_wav_int16, save_jsonl
from ..tools.util.splits import assign_splits_by_group

def main():
    ap = argparse.ArgumentParser(description='Build 5–15s dataset with QC, hum-notch, speaker filter, subtitles')
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sr', type=int, default=24000)
    ap.add_argument('--min_sec', type=float, default=5.0)
    ap.add_argument('--max_sec', type=float, default=15.0)
    ap.add_argument('--margin_ms', type=int, default=150)
    ap.add_argument('--hum_autonotch', action='store_true')
    ap.add_argument('--mains', type=int, choices=[50,60], default=None)
    ap.add_argument('--notch_strength', type=float, default=0.7)
    ap.add_argument('--target_speaker_dir', type=str, default=None)
    ap.add_argument('--spk_sim_thresh', type=float, default=0.5)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir); ensure_dir(out_root)
    audio_exts = {'.wav','.mp3','.flac','.m4a','.aac','.ogg'}
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in audio_exts])
    if not files: raise SystemExit(f"No audio in {in_dir}")

    spk = SpeakerFilter(Path(args.target_speaker_dir) if args.target_speaker_dir else None,
                        sr=args.sr, thresh=args.spk_sim_thresh)

    all_items = []; wav_cache = {}
    for p in tqdm(files, desc='Segmenting'):
        clips, y = segment_file(p, sr=args.sr, min_sec=args.min_sec, max_sec=args.max_sec,
                                margin_ms=args.margin_ms, mains_hz=args.mains,
                                apply_notch=args.hum_autonotch, notch_strength=args.notch_strength,
                                speaker_filter=spk)
        wav_cache[p] = (y, args.sr)
        for c in clips:
            all_items.append({
                "doc_id": c.doc_id,
                "start_sec": round(c.start_s,3),
                "end_sec": round(c.end_s,3),
                "text": c.text or "",
                "qc": c.qc,
                "source": str(p.name)
            })
    if not all_items: raise SystemExit('No clips produced.')

    buckets = assign_splits_by_group(all_items, key_fn=lambda x: (x['doc_id']))

    for split, items in buckets.items():
        sdir = out_root / split; ensure_dir(sdir); meta = []
        for i, it in enumerate(items, start=1):
            base = it['doc_id']
            fname = f"{base}_{i:04d}.wav" if split=='train' else (f"{base}_v_{i:04d}.wav" if split=='val' else f"{base}_t_{i:04d}.wav")
            y, sr = wav_cache[[p for p in wav_cache if Path(p).name==it['source']][0]]
            s = int(it['start_sec']*sr); e = int(it['end_sec']*sr)
            from ..tools.audio.qc import trim_with_margin
            clip = trim_with_margin(y[s:e], sr, margin_ms=args.margin_ms)
            out_wav = sdir / fname
            write_wav_int16(out_wav, clip, sr)
            meta.append({
                "audio": str(out_wav.relative_to(out_root)),
                "text": it['text'],
                "doc_id": it['doc_id'],
                "start_sec": it['start_sec'],
                "end_sec": it['end_sec'],
                "duration_sec": round((len(clip)/sr),3),
                "sample_rate": sr, "bit_depth": 16, "num_channels": 1,
                "qc": it['qc']
            })
        save_jsonl(sdir/'metadata.jsonl', meta)

    (out_root/'run_summary.json').write_text(json.dumps({
        "num_files": len(files),
        "splits": {k: len(v) for k,v in buckets.items()},
        "config": vars(args)
    }, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()

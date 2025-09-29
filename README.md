# Voice Podcast Template (Sesame/CSM-ready)

This repo gives you a clean starting point to record, segment, and prepare your voice dataset for fine-tuning a TTS model (e.g., **Sesame CSM-1B** with a LoRA adapter).

## Folder layout
```
voice-podcast-template/
  data/
    train/         # utterance WAVs + metadata.csv
    val/           # validation utterances + metadata.csv
    test/          # held-out evaluation utterances + metadata.csv
    ref/           # a few 1–3s "anchor" clips you like for inference
  scripts/
    segment_and_prep.py  # VAD-based splitter + (optional) ASR transcript
```
- All audio should be **mono, 24 kHz, 16-bit PCM WAV**.
- `metadata.csv` format (pipe-separated): `rel/path.wav|Exact transcript with punctuation.`

## Quick start
1. Record long takes (same mic/room).
2. Run the segmentation script to auto-split into **5–15s** utterances, loudness-normalize, and create metadata files. You can optionally enable **Whisper**/**faster-whisper** to pre-fill transcripts.
3. Manually spot-fix transcripts for accuracy and style (spell out numbers, expand abbreviations).
4. Fine-tune your model and iterate.

## Segmentation script (key features)
- **VAD-based** splitting (WebRTC VAD) with smart merging and length caps.
- **Target loudness** (default `-20 LUFS`) using `pyloudnorm`.
- **Resample** to 24 kHz and write **16-bit PCM** WAV.
- **Optional ASR**: `--whisper` or `--faster-whisper` to auto-fill transcripts.
- **Train/val/test split** that **keeps utterances from the same source take together** to reduce leakage.
- **Ref clips**: automatically exports a few clean 1–3s anchors to `data/ref/`.

## Example usage
```
# Create from a folder of long recordings (WAV/FLAC/MP3 etc.)
python scripts/segment_and_prep.py   --inputs /path/to/long_takes   --outdir ./data   --lufs -20   --min-sec 5 --max-sec 15   --vad 2   --whisper base  # or: --faster-whisper base
```

## Dependencies
Install with pip (choose **one** ASR option or none):
```
pip install soundfile numpy librosa pyloudnorm webrtcvad tqdm argostranslate==1.9.6
# ASR options (optional):
pip install -U openai-whisper
# OR
pip install -U faster-whisper
```
> If you use `faster-whisper`, make sure your FFmpeg is available (`ffmpeg -version`).

## Metadata format
- `data/train/metadata.csv` and `data/val/metadata.csv` contain lines like:
```
you_000123.wav|This is how I'd like this to be read.
```
- Keep punctuation you want reflected in prosody.
- Use an **80/10/10** split (the script does this by source take).

## Tips
- Record in long takes if that’s easier; the script will split on voice activity.
- Keep environment consistent (same mic, gain, room, distance).
- Avoid heavy processing; mild noise reduction is okay if it doesn’t hurt timbre.
- Include your **podcast style**: intro, newsy read, conversational asides, emphatic lines, lists, dates/URLs you plan to read.

# voice-dataset-kit

Segment long-form audio into 5–15s 24kHz mono PCM WAV clips, with QC (hum/plosive/clipping), speaker filtering, subtitle support, and a tiny reviewer.

## Install

```shell
uv sync
```

## Build dataset

```shell
uv run python -m cli.build_dataset   --in_dir raw   --out_dir data   --sr 24000 --min_sec 5 --max_sec 15   --margin_ms 150   --hum_autonotch --mains 60 --notch_strength 0.7   --target_speaker_dir data/ref   --spk_sim_thresh 0.55
```

## Review

```shell
uv run python -m cli.reviewer --out_dir data
# open http://127.0.0.1:7860
```

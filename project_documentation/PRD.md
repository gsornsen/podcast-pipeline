# Product Requirements Document (PRD)

## 0) Document Info

* **Product**: Voice Dataset Kit → Speaker Model Training → Podcast Synthesis
* **Owner**: [You]
* **Version**: v1.0 (Sept/Oct 2025)
* **Related folder**: `voice-dataset-kit` (multi-file toolchain created), model-training repo (to be created), podcast-synthesis repo (to be created)

---

## 1) Problem & Goals

### Problem

We need an end‑to‑end pipeline that:

1. Converts long-form recordings into clean ~5–15 s utterances with exact transcripts and quality control.
2. Trains a high‑quality voice model that runs on a single RTX 4090.
3. Uses that model to synthesize full podcast episodes from a script—first with one voice, then with 2–4 voices.

### Goals (what success looks like)

* **G1 — Data quality**: Automatic segmentation with strong QC, exact text match, and easy human review.
* **G2 — Model quality**: Naturalness and identity preservation competitive with current open models, optimized for a 4090.
* **G3 — Productionization**: Deterministic CLI/API that can generate a 10-30 min episode reliably with reproducible settings.

### Non-goals (for now)

* Live, real‑time streaming TTS; voice conversion from arbitrary sources; multilingual training (unless explicitly added later).

---

## 2) Personas & Use Cases

* **Producer/Host (primary)**: Prepares dataset, trains own voice model, renders weekly podcast from scripts.
* **Audio Editor (secondary)**: Uses reviewer UI to fix transcripts, reject noisy clips, and spot‑check renders.
* **ML Engineer (secondary)**: Tunes LoRA hyperparameters and evaluates model regressions.

Key use cases:

1. **Prepare dataset** from a folder of long takes + optional SRT/VTT to utterances + `metadata.jsonl`.
2. **Train voice model** (LoRA over a base checkpoint) with the dataset.
3. **Render episode** from a transcript using the trained model(s) with anchors, loudness, pacing, and export.

---

## 3) Scope

### In Scope (Milestone‑by‑Milestone)

* **M0 (complete in tooling)**: Segmentation tool with VAD, subtitle merge, hum/plosive QC, auto‑notch, speaker filter, reviewer.
* **M1 (single speaker)**: Train a personal voice model and render a full episode from a script.
* **M2 (multi‑speaker)**: Render 2–4 speakers using 2–4 trained models, including dialogue orchestration and mixing.

### Out of Scope (initially)

* ASR‑based automatic transcript correction; music composition; sound‑design automation.

---

## 4) Functional Requirements

### 4.0 General CLI Requirements

**Usability**

* Console-entry points (voicekit-built, voicekit-review, etc) so that cli entrypoints can be used without requiring uv run commands.
* Modern and user friendly cli/tui for users not familiar with passing in complex/long lists of cli args

### 4.1 Data Preparation (CLI: `cli.build_dataset`)

**Inputs**

* Directory of long audio files (`.wav/.mp3/.flac/.m4a/.aac/.ogg`).
* Optional sidecars: `.srt`/`.vtt` (subtitle cues), `.txt` (paragraphs).
* Optional target speaker reference directory (`data/ref/*.wav`).

**Processing**

* Resample to **24,000 Hz**, mono; 16‑bit PCM on output.
* Segment by:
  * Prefer SRT/VTT (non‑monotonic reconciliation → merged windows ~5–15 s).
  * Fallback VAD (Silero) with merged speech regions → trimmed utterances.
* **QC**: RMS floor, SNR, clipping detection; **hum detector** (50/60 Hz harmonics); **plosive detector** (LF bursts).
* **Auto‑notch hum** (configurable strength), optional.
* **Speaker filter**: ECAPA embeddings, cosine similarity; threshold exposed.
* **Trim**: keep ~100-200 ms leading/trailing silence.
* **Split**: deterministic 80/10/10 by document (and paragraph when available).

**Outputs**

* Dataset layout:

  * `data/{train,val,test}/<doc>_{####|v_####|t_####}.wav`
  * `data/{train,val,test}/metadata.jsonl` with fields: `audio`, `text`, `doc_id`, `start_sec`, `end_sec`, `duration_sec`, `sample_rate`, `bit_depth`, `num_channels`, `qc{rms_db,snr_db,clipped,hum_score,plosive_ratio}`.
  * `data/ref/` (anchors the user provides).
  * `data/quarantine/` (reviewer moves rejected files here).
* `run_summary.json` with counts and config.

**Reviewer (CLI: `cli.reviewer`)**

* Audition clip audio, edit transcript inline, **Reject** to quarantine.
* Writes updated `metadata.jsonl`; preserves relative paths when quarantining.

**Acceptance Criteria**

* ≥ **90%** of clips between 5–15 s.
* Reviewer UX allows edit+reject at ≥ **60 clips/min** on a standard laptop.
* Post‑review, ≤ **5%** clips contain obvious defects (audible hum, plosive pop, clipping).

### 4.2 Model Training (M1: Single Speaker)

**Default model**: **Sesame CSM‑1B** with LoRA speaker adaptation (target; runs on 24 GB RTX 4090). Alternatives for experimentation: XTTS‑v2, CosyVoice‑2 small, Fish‑Speech/OpenAudio mini.

**Training Data Requirements**

* **30–90 minutes** curated utterances (start with 30–60 min).
* Exact transcript per utterance; punctuation preserved.
* Same mic/chain (ideally) as intended for inference.

**Training Configuration (reference)**

* Mixed precision **bf16/fp16**; gradient checkpointing.
* LoRA ranks 8–16; target blocks: audio decoder + selected backbone layers (exact mapping per repo).
* Batch 2–4, grad accumulation 4–8; learning rate warmup + cosine decay.
* Early stop on validation WER/CER and listening tests.

**Artifacts**

* LoRA adapter weights + YAML config (tokenizer/audio preproc, normalization, sampling);
* Inference script accepting `--anchor` (1–3 s) to stabilize identity.

**Acceptance Criteria**

* **Naturalness** (MOS‑lite subjective panel): **≥ 4.0 / 5.0** on held‑out script.
* **Intelligibility**: WER ≤ **7%**, CER ≤ **3%** vs. ground‑truth renders of held‑out text.
* **Identity similarity**: Speaker embedding cosine ≥ **0.75** (ECAPA vs. reference anchors).
* **Throughput**: Generate a **30‑minute** mono episode in **≤ 30 minutes** on RTX 4090.

### 4.3 Podcast Synthesis (M1 → single voice)

**Inputs**: Episode script (Markdown/JSON), optional SSML‑lite tags for pauses/emphasis, `ref/` anchors.

**Rendering Pipeline**

1. Convert script → utterance list; insert anchors per section/paragraph.
2. Synthesize audio with trained model; batch by paragraphs.
3. Post‑processing: de‑ess (light), loudness normalize to **−19 LUFS (mono)**, room tone bed optional.
4. Export WAV and episode JSON (timestamps per paragraph/line).

**Acceptance Criteria**

* End‑to‑end deterministic render (same config → same audio).
* Audible continuity: no sudden timbre drift across sections.

### 4.4 Podcast Synthesis (M2 → 2–4 speakers)

**Additional Inputs**: Role → model mapping (e.g., `Host`=you, `GuestA/B/C` = trained voices), dialogue script with role tags.

**Rendering Pipeline Additions**

* Assign each line to a model; per‑speaker anchors.
* Optional **pacing model**: inter‑line pauses, overlaps capped, crossfades.
* **Mixing**: ducking on interjections, stereo scene preset (L/R pan small for separation), music bed support.

**Acceptance Criteria**

* 2–4‑speaker episode renders without clipping; crosstalk ≤ **5%** of lines.
* Dialogue timing natural per listening panel; role identity separable by blind test ≥ **80%** accuracy.

---

## 5) Non‑Functional Requirements

* **Reproducibility**: All CLIs accept a `--seed`; configs saved alongside outputs.
* **Performance**: Dataset build processes **≥ 1 hour of audio per 5 minutes** on a 16‑core desktop (no ASR).
* **Observability**: `run_summary.json` + progress bars + optional verbose QC logging.
* **Portability**: Linux; single‑GPU (4090) training default.

---

## 6) APIs & CLIs

### Data Prep

```bash
uv run python -m cli.build_dataset \
  --in_dir raw --out_dir data \
  --sr 24000 --min_sec 5 --max_sec 15 --margin_ms 150 \
  --hum_autonotch --mains 60 --notch_strength 0.7 \
  --target_speaker_dir data/ref --spk_sim_thresh 0.55
```

Reviewer:

```bash
uv run ython -m cli.reviewer --out_dir data
```

### Training (to be implemented)

* `train_single.py` — loads dataset JSONL, builds dataloaders, applies LoRA to base checkpoint, trains and exports adapter.
* `infer_single.py` — render text/anchors to audio.
* Config via `yaml` (paths, model name, LoRA ranks, LR schedule, batch/accum, eval sets).

### Synthesis

* `synthesize_podcast.py` (M1): single‑speaker.
* `synthesize_podcast_multi.py` (M2): multi‑speaker with role assignment.

---

## 7) Data Contracts & Formats

* **`metadata.jsonl`** (per split):

```json
{
  "audio": "train/you_0001.wav",
  "text": "Exact transcript.",
  "doc_id": "you",
  "start_sec": 12.345,
  "end_sec": 24.567,
  "duration_sec": 12.222,
  "sample_rate": 24000,
  "bit_depth": 16,
  "num_channels": 1,
  "qc": {"rms_db": -20.3, "snr_db": 18.7, "clipped": 0.0, "hum_score": 0.08, "plosive_ratio": 0.00}
}
```

* **`episode.json`** (render output): list of sections/paragraphs with timestamps, speaker labels (M2), and final WAV path.

---

## 8) Evaluation Plan

* **Objective**: WER/CER on held‑out text using ASR back‑eval (Whisper small/base). Target ≤ 7% / 3%.
* **Subjective**: 10‑listener MOS‑lite (1–5) on 10 held‑out prompts: target ≥ 4.0.
* **Identity**: ECAPA cosine sim vs. anchors: ≥ 0.75.
* **Render stability**: No more than 1 audible glitch per 10 minutes (panel annotated).

---

## 9) Risks & Mitigations

* **Risk**: Overfitting to anchors → robotic delivery.
  * *Mitigation*: Data variety (prosody), early stopping on MOS regressions.
* **Risk**: Subtitle timing drift.
  * *Mitigation*: Non‑monotonic merge + reviewer spot checks.
* **Risk**: Hum/plosives not fully removed.
  * *Mitigation*: Auto‑notch (gentle), flag high scores for manual review.
* **Risk**: Legal/consent.
  * *Mitigation*: Explicit consent logging; train only on own/authorized voice.

---

## 10) Rollout & Milestones

* **M0 — Data Prep v1.1** *(done)*: multi‑file toolkit, reviewer, hum/plosive, speaker filter, auto‑notch.
* **M1 — Single‑Speaker Training & Synthesis** *(4–6 weeks)*:
  * T1: Training scripts + YAML, first LoRA trained, eval harness.
  * T2: Inference CLI with anchors, episode synth tool.
  * Exit criteria: meets M1 acceptance criteria.
* **M2 — Multi‑Speaker (2–4 voices)** *(4–6 weeks after M1)*:
  * T1: Train/ingest additional speaker adapters, role assignment.
  * T2: Mixer (ducking/crossfade), pan presets, music bed track.
  * Exit criteria: meets M2 acceptance criteria.

---

## 11) Future Work (Backlog)

* SSML‑like control tokens (rate, pitch, emphasis) and inline **[sfx: ...]** cues.
* Automatic generation of sound effects, music, background noises that improve the podcast and drive engagement
* Streaming synthesis; real‑time agent glue.
* Automatic prosody alignment to reference takes.
* Web UI for end‑to‑end generation and review.

---

## 12) Appendix — Ops & Settings

* **Loudness targets**: −19 LUFS (mono), true‑peak ≤ −1 dBTP.
* **File formats**: WAV 24 kHz PCM 16‑bit for both dataset and renders.
* **Seeds**: Default 42; include seed hashing into output manifest.
* **Logging**: Train/val metrics → CSV + TensorBoard; dataset `run_summary.json`.

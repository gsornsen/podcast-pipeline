# Technical Design Document (TDD)

## 0) Document Info

* **Product**: Voice Dataset Kit → Speaker Model Training → Podcast Synthesis Pipeline
* **Version**: v1.0 (Sept/Oct 2025)
* **Architecture**: Modular Python toolchain with PyTorch-based model training
* **Target Hardware**: Single RTX 4090 (24GB VRAM) for training and inference
* **Related Components**:
  * `voice-dataset-kit/` - Data preparation and QC tools (✅ Implemented)
  * `model-training/` - LoRA fine-tuning pipeline (To be created)
  * `podcast-synthesis/` - Episode generation and mixing (To be created)

---

## 1) Executive Summary

This document defines the technical architecture and implementation plan for an end-to-end voice cloning and podcast synthesis pipeline. The system takes long-form recordings, processes them into high-quality training datasets, trains personalized voice models using LoRA adaptation, and synthesizes full podcast episodes with up to 4 speakers.

### Key Technical Decisions

* **Model**: Sesame CSM-1B with LoRA adaptation (optimized for RTX 4090)
* **Audio Format**: 24kHz, 16-bit PCM, mono throughout pipeline
* **Framework**: PyTorch 2.8+ with mixed precision training
* **CLI-First**: All components accessible via modern CLI/TUI interfaces
* **Deterministic**: Seed-controlled generation for reproducible outputs

---

## 2) System Architecture

### 2.1 High-Level Architecture

```shell
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Long Recordings │ Subtitles (SRT/VTT) │ Reference Anchors      │
└──────────┬──────────────┬─────────────────────┬─────────────────┘
           │              │                     │
           ▼              ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  • VAD Segmentation     • Subtitle Alignment                    │
│  • QC Pipeline          • Speaker Filtering                     │
│  • Hum/Plosive Detection• Auto-notch Filtering                  │
│  • Dataset Splitting    • Reviewer UI                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  • Sesame CSM-1B Base   • LoRA Adaptation (rank 8-16)           │
│  • Mixed Precision bf16 • Gradient Checkpointing                │
│  • ECAPA Verification   • WER/MOS Evaluation                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYNTHESIS LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  • Script Parsing       • Anchor Injection                      │
│  • Batch Generation     • Multi-speaker Orchestration           │
│  • Loudness Normalization• Mixing & Ducking                     │
│  • Export Pipeline      • Timestamp Alignment                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Dependencies

```python
# Core Dependencies
pytorch >= 2.8.0          # Deep learning framework
torchaudio >= 2.8.0       # Audio processing
librosa >= 0.11.0         # Audio analysis
soundfile >= 0.13.1       # Audio I/O
scipy >= 1.16.2           # Signal processing

# Model-specific
speechbrain >= 1.0.3      # Speaker embeddings (ECAPA)
transformers >= 4.40.0    # Model architecture
accelerate >= 0.30.0      # Training optimization
peft >= 0.10.0           # LoRA implementation

# Audio Processing
pyloudnorm >= 0.1.1       # LUFS normalization
webrtcvad >= 2.0.10       # Voice activity detection
pysrt >= 1.1.2            # Subtitle parsing
webvtt-py >= 0.5.1        # WebVTT support

# Development Tools
ruff >= 0.13.2            # Linting
mypy >= 1.18.2            # Type checking
pre-commit >= 4.3.0       # Git hooks
```

---

## 3) Technical Requirements

### 3.1 Audio Processing Pipeline

```python
# Audio Processing Configuration
AUDIO_CONFIG = {
    "sample_rate": 24000,      # Hz
    "bit_depth": 16,           # bits
    "channels": 1,             # mono
    "format": "PCM_16",        # Linear PCM
    "loudness_target": -19.0,  # LUFS (mono)
    "true_peak": -1.0,         # dBTP
    "segment_length": (5, 15), # seconds (min, max)
    "margin_ms": 150,          # silence padding
}

# Quality Control Thresholds
QC_THRESHOLDS = {
    "rms_floor": -40.0,        # dB
    "snr_min": 5.0,            # dB
    "clipping_thresh": 0.999,
    "hum_score_max": 0.3,      # 0-1 scale
    "plosive_ratio_max": 0.05, # ratio of frames
}
```

### 3.2 Model Training Configuration

```python
# LoRA Training Configuration
TRAINING_CONFIG = {
    "base_model": "sesame-csm-1b",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": [
        "audio_decoder.layers.*",
        "text_encoder.layers.10.*",
        "text_encoder.layers.11.*",
    ],

    "batch_size": 2,
    "gradient_accumulation": 8,
    "effective_batch": 16,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,

    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "max_steps": 2000,
    "eval_steps": 100,
    "save_steps": 200,

    "early_stopping_patience": 5,
    "early_stopping_metric": "eval_wer",
}
```

### 3.3 Performance Requirements

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Dataset Processing | ≥1 hour audio / 5 min | Wall clock time |
| Training Throughput | ≥100 steps/hour | On RTX 4090 |
| Inference Speed | ≥1.0x realtime | 30min episode ≤ 30min |
| Memory Usage | ≤22GB VRAM | nvidia-smi peak |
| Storage per Voice | ≤500MB | LoRA adapter + config |

---

## 4) Implementation Phases

### Phase 0: Data Preparation [✅ COMPLETE]

**Status**: Implemented in `voice-dataset-kit/`

Key components already built:

* VAD-based segmentation with Silero
* Subtitle alignment and merging
* Hum detection and auto-notch filtering
* Plosive detection
* Speaker filtering with ECAPA embeddings
* Flask-based reviewer UI
* Deterministic train/val/test splitting

### Phase 1: Model Training Infrastructure [4 weeks]

#### Week 1-2: Training Pipeline

```python
# model_training/train_lora.py
class VoiceLoRATrainer:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.model = self._load_base_model()
        self.lora_config = self._setup_lora()

    def _load_base_model(self):
        # Load Sesame CSM-1B with memory optimization
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "sesame/csm-1b",
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
        )
        model.gradient_checkpointing_enable()
        return model

    def _setup_lora(self):
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=self.config["target_modules"],
            lora_dropout=self.config["lora_dropout"],
        )
        return get_peft_model(self.model, config)

    def train(self, dataset_path: str):
        # Training loop with mixed precision
        # Early stopping on validation metrics
        # Save best checkpoint based on MOS/WER
        pass
```

#### Week 2-3: Evaluation Framework

```python
# model_training/evaluate.py
class VoiceEvaluator:
    def __init__(self, model_path: str, ref_dir: str):
        self.model = load_model(model_path)
        self.references = load_anchors(ref_dir)
        self.whisper = load_whisper_for_eval()
        self.ecapa = load_ecapa()

    def compute_wer(self, text_prompts: List[str]) -> float:
        """Generate audio and compute WER using Whisper"""

    def compute_speaker_similarity(self, generated: np.ndarray) -> float:
        """Compare ECAPA embeddings with reference"""

    def compute_mos_proxy(self, generated: np.ndarray) -> float:
        """Compute objective MOS approximation"""
```

#### Week 3-4: CLI and Configuration

```bash
# Training CLI
uv run train-voice \
    --dataset data/ \
    --config configs/lora_default.yaml \
    --output models/voice_v1/ \
    --eval-every 100 \
    --seed 42

# Evaluation CLI
uv run evaluate-voice \
    --model models/voice_v1/best.pt \
    --test-set data/test/ \
    --metrics wer,similarity,mos
```

### Phase 2: Single-Speaker Synthesis [3 weeks]

#### Week 1: Core Inference Pipeline

```python
# podcast_synthesis/synthesize.py
class PodcastSynthesizer:
    def __init__(self, model_path: str, config: dict):
        self.model = load_lora_model(model_path)
        self.anchors = load_anchors(config["ref_dir"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def synthesize_utterance(
        self,
        text: str,
        anchor_idx: int = 0,
        seed: int = 42
    ) -> np.ndarray:
        """Generate single utterance with anchor conditioning"""
        torch.manual_seed(seed)

        # Prepare inputs
        anchor = self.anchors[anchor_idx]
        text_tokens = self.tokenize(text)

        # Generate with temperature control
        with torch.no_grad():
            audio = self.model.generate(
                text_tokens,
                speaker_embedding=anchor,
                max_length=self.config["max_length"],
                temperature=0.7,
            )
        return audio.cpu().numpy()

    def render_episode(self, script_path: str) -> Tuple[np.ndarray, dict]:
        """Render full episode from script"""
        script = load_script(script_path)
        segments = []
        timestamps = []

        for paragraph in script.paragraphs:
            # Insert anchor at paragraph boundaries
            audio = self.synthesize_utterance(
                paragraph.text,
                anchor_idx=paragraph.anchor_idx
            )
            segments.append(audio)
            timestamps.append({
                "start": sum(len(s)/SR for s in segments[:-1]),
                "end": sum(len(s)/SR for s in segments),
                "text": paragraph.text
            })

        # Concatenate with crossfade
        full_audio = crossfade_segments(segments)

        # Post-processing
        full_audio = loudness_normalize(full_audio, -19.0)

        return full_audio, {"timestamps": timestamps}
```

#### Week 2: Script Parsing and SSML Support

```python
# podcast_synthesis/script_parser.py
class ScriptParser:
    def parse_markdown(self, path: str) -> Episode:
        """Parse markdown with SSML-lite tags"""
        # Support for:
        # <pause ms="500"/>
        # <emphasis level="strong">text</emphasis>
        # <prosody rate="slow">text</prosody>

    def parse_json(self, path: str) -> Episode:
        """Parse structured JSON episode format"""
```

#### Week 3: Post-processing Pipeline

```python
# podcast_synthesis/postprocess.py
def apply_de_essing(audio: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Gentle de-essing for sibilance control"""

def add_room_tone(audio: np.ndarray, level_db: float = -50) -> np.ndarray:
    """Add subtle room tone between segments"""

def export_episode(
    audio: np.ndarray,
    metadata: dict,
    output_dir: str,
    formats: List[str] = ["wav", "mp3"]
):
    """Export with multiple format support"""
```

### Phase 3: Multi-Speaker Synthesis [4 weeks]

#### Week 1-2: Multi-Model Management

```python
# podcast_synthesis/multi_speaker.py
class MultiSpeakerSynthesizer:
    def __init__(self, speaker_configs: Dict[str, dict]):
        self.speakers = {}
        for name, config in speaker_configs.items():
            self.speakers[name] = PodcastSynthesizer(
                config["model_path"],
                config["settings"]
            )

    def render_dialogue(self, script: DialogueScript) -> np.ndarray:
        """Render multi-speaker dialogue with timing"""
        tracks = defaultdict(list)

        for line in script.lines:
            speaker = self.speakers[line.speaker]
            audio = speaker.synthesize_utterance(line.text)
            tracks[line.speaker].append({
                "audio": audio,
                "start_time": line.start_time,
                "end_time": line.end_time
            })

        # Mix tracks with ducking
        mixed = self.mix_tracks(tracks)
        return mixed
```

#### Week 2-3: Dialogue Orchestration

```python
# podcast_synthesis/orchestration.py
class DialogueOrchestrator:
    def calculate_timing(self, script: DialogueScript) -> List[TimedLine]:
        """Calculate natural timing with overlaps and pauses"""

    def apply_ducking(
        self,
        primary: np.ndarray,
        secondary: np.ndarray,
        duck_db: float = -6.0
    ) -> np.ndarray:
        """Duck secondary when primary is speaking"""

    def create_spatial_scene(
        self,
        speakers: Dict[str, np.ndarray],
        positions: Dict[str, float]  # -1.0 to 1.0 (L to R)
    ) -> np.ndarray:
        """Create stereo scene with speaker positioning"""
```

#### Week 3-4: Advanced Features

```python
# Music bed support
def add_music_bed(
    dialogue: np.ndarray,
    music_path: str,
    duck_amount: float = 0.3
) -> np.ndarray:
    """Add background music with auto-ducking"""

# Effect injection
def inject_effects(
    audio: np.ndarray,
    effects: List[Effect],
    timestamps: List[float]
) -> np.ndarray:
    """Insert sound effects at specified points"""
```

---

## 5) API Specifications

### 5.1 CLI Interfaces

```bash
# Dataset preparation (existing)
voicekit-build \
    --in-dir raw_recordings/ \
    --out-dir data/ \
    --config configs/dataset.yaml

# Model training
voicekit-train \
    --dataset data/ \
    --base-model sesame-csm-1b \
    --output models/my_voice/ \
    --config configs/training.yaml

# Single-speaker synthesis
voicekit-synth \
    --model models/my_voice/best.pt \
    --script episodes/ep001.md \
    --output episodes/ep001.wav \
    --seed 42

# Multi-speaker synthesis
voicekit-synth-multi \
    --speakers config/speakers.yaml \
    --script episodes/dialogue.json \
    --output episodes/multi.wav \
    --mix-config configs/mixing.yaml
```

### 5.2 Configuration Formats

#### Training Configuration (YAML)

```yaml
# configs/training.yaml
model:
  base: "sesame-csm-1b"
  dtype: "bfloat16"

lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - "decoder.layers.*"

training:
  batch_size: 2
  gradient_accumulation: 8
  learning_rate: 5e-5
  max_steps: 2000
  warmup_steps: 100

evaluation:
  eval_steps: 100
  metrics: ["wer", "similarity", "mos_proxy"]
  early_stopping:
    patience: 5
    metric: "wer"
    mode: "min"
```

#### Episode Script (JSON)

```json
{
  "episode": {
    "title": "Tech News Weekly",
    "speakers": {
      "host": "models/host_voice.pt",
      "guest": "models/guest_voice.pt"
    },
    "segments": [
      {
        "type": "monologue",
        "speaker": "host",
        "text": "Welcome to Tech News Weekly...",
        "anchor_idx": 0
      },
      {
        "type": "dialogue",
        "lines": [
          {"speaker": "host", "text": "What do you think about..."},
          {"speaker": "guest", "text": "That's a great question..."}
        ]
      }
    ]
  }
}
```

### 5.3 Output Formats

#### Episode Manifest

```json
{
  "episode_id": "ep001_2025_09_28",
  "duration_sec": 1800.5,
  "speakers": ["host"],
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 45.2,
      "text": "Welcome to the show...",
      "speaker": "host"
    }
  ],
  "audio_files": {
    "wav": "episodes/ep001.wav",
    "mp3": "episodes/ep001.mp3"
  },
  "generation": {
    "model": "models/host_v2.pt",
    "seed": 42,
    "timestamp": "2025-09-28T10:30:00Z"
  }
}
```

---

## 6) Testing & Evaluation Strategy

### 6.1 Unit Testing

```python
# tests/test_training.py
def test_lora_initialization():
    """Verify LoRA layers are correctly initialized"""

def test_gradient_accumulation():
    """Ensure effective batch size is correct"""

def test_checkpoint_saving():
    """Verify model checkpoints are saveable/loadable"""
```

### 6.2 Integration Testing

```python
# tests/test_pipeline.py
def test_end_to_end_single_speaker():
    """Test full pipeline from dataset to synthesized audio"""

def test_deterministic_generation():
    """Verify same seed produces identical output"""

def test_multi_speaker_mixing():
    """Test dialogue generation with 2+ speakers"""
```

### 6.3 Performance Testing

```python
# tests/test_performance.py
def test_inference_throughput():
    """Measure generation speed vs realtime"""

def test_memory_usage():
    """Monitor VRAM usage during training/inference"""

def test_batch_processing():
    """Verify batch synthesis maintains quality"""
```

### 6.4 Quality Metrics

| Metric | Target | Test Method |
|--------|--------|-------------|
| WER | ≤7% | Whisper transcription of generated audio |
| CER | ≤3% | Character-level accuracy |
| Speaker Similarity | ≥0.75 | ECAPA cosine similarity |
| MOS (subjective) | ≥4.0/5.0 | Human evaluation panel |
| Latency | <100ms | First token generation time |

---

## 7) Implementation Sequence

### Sprint 0: Environment Setup

* [ ] Create model-training/ directory structure
* [ ] Set up PyTorch environment with CUDA
* [ ] Install and test Sesame CSM-1B base model
* [ ] Verify RTX 4090 memory and compute capabilities

### Sprint 1: Basic Training

* [ ] Implement data loader for JSONL format
* [ ] Create LoRA trainer class
* [ ] Add gradient accumulation and mixed precision
* [ ] Implement basic evaluation (WER only)
* [ ] **Deliverable**: First LoRA model trained on sample data

### Sprint 2: Evaluation Suite

* [ ] Add ECAPA speaker similarity
* [ ] Implement MOS proxy metrics
* [ ] Create evaluation report generator
* [ ] Add early stopping logic
* [ ] **Deliverable**: Training script with full metrics

### Sprint 3: Single-Speaker Synthesis

* [ ] Implement basic inference pipeline
* [ ] Add anchor conditioning
* [ ] Create script parser (Markdown)
* [ ] Implement loudness normalization
* [ ] **Deliverable**: First generated podcast episode

### Sprint 4: Quality Improvements

* [ ] Add de-essing and post-processing
* [ ] Implement crossfade between segments
* [ ] Add room tone generation
* [ ] Optimize inference speed
* [ ] **Deliverable**: Production-quality single voice

### Sprint 5: Multi-Speaker Foundation

* [ ] Create multi-model manager
* [ ] Implement dialogue parser
* [ ] Add basic mixing (no ducking)
* [ ] Test with 2 speakers
* [ ] **Deliverable**: Two-speaker dialogue demo

### Sprint 6: Advanced Mixing

* [ ] Implement audio ducking
* [ ] Add stereo positioning
* [ ] Create timing orchestrator
* [ ] Add music bed support
* [ ] **Deliverable**: Full multi-speaker episode

### Sprint 7: Polish & Optimization

* [ ] Performance optimization
* [ ] Error handling and recovery
* [ ] Documentation and examples
* [ ] Final testing and validation
* [ ] **Deliverable**: Production-ready system

---

## 8) Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model overfitting | Poor generalization | Data augmentation, early stopping, dropout |
| Memory overflow | Training failure | Gradient checkpointing, batch size tuning |
| Timing drift | Unnatural dialogue | Explicit timestamp management, overlap control |
| Quality regression | User dissatisfaction | Automated testing, A/B comparison |
| Inference slowdown | Poor UX | Caching, batch processing, model quantization |

### Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data privacy | Legal issues | Local processing only, no cloud uploads |
| Model distribution | Large file sizes | LoRA adapters only (< 500MB) |
| Dependency conflicts | Setup failures | Pinned versions, Docker containers |
| Hardware limitations | Cannot train/run | Cloud GPU fallback options |

---

## 9) Success Metrics

### Milestone 1 (Single Speaker)

* [x] Dataset preparation tool complete
* [ ] LoRA training converges in < 2000 steps
* [ ] WER < 7% on test set
* [ ] Generate 10-minute episode < 10 minutes
* [ ] MOS ≥ 4.0 from test panel

### Milestone 2 (Multi-Speaker)

* [ ] Support 2-4 concurrent speakers
* [ ] Dialogue timing feels natural
* [ ] Speaker identity separable > 80%
* [ ] No clipping or artifacts in mix
* [ ] 30-minute episode < 30 minutes

### Final Acceptance

* [ ] End-to-end pipeline documented
* [ ] All CLI tools have help text
* [ ] Test coverage > 80%
* [ ] Performance meets all targets
* [ ] 5 complete episodes generated successfully

---

## 10) Future Enhancements

### Phase 4 (Backlog)

* Streaming synthesis for live applications
* SSML full compliance
* Automatic prosody transfer from reference
* Emotion control tokens
* Web UI for non-technical users
* Cloud training pipeline
* Model compression/quantization
* Mobile inference support

### Research Opportunities

* Few-shot voice adaptation (< 5 minutes data)
* Cross-lingual voice transfer
* Real-time voice conversion
* Music and SFX generation
* Automatic script generation from topics

---

## Appendix A: Code Examples

### A.1 Training Script Structure

```python
#!/usr/bin/env python3
"""
Train LoRA adapter for voice cloning.
Usage: python train_lora.py --config config.yaml
"""

import argparse
import torch
from pathlib import Path
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSpeechSeq2Seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=8,
    )

    # Load configuration
    config = load_yaml(args.config)

    # Setup model with LoRA
    model = setup_model_with_lora(config)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        args.dataset,
        config["batch_size"]
    )

    # Training loop
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output,
        accelerator=accelerator,
    )

if __name__ == "__main__":
    main()
```

### A.2 Inference Pipeline

```python
#!/usr/bin/env python3
"""
Generate podcast episode from script.
Usage: python synthesize.py --model model.pt --script episode.md --output episode.wav
"""

import torch
import numpy as np
from pathlib import Path

class EpisodeGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        self.anchors = self.load_anchors(model_path)

    def load_model(self, path: str):
        """Load LoRA-adapted model"""
        checkpoint = torch.load(path, map_location=self.device)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint["base_model"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        # Apply LoRA weights
        model = PeftModel.from_pretrained(model, path)
        model.eval()
        return model

    def generate_episode(self, script_path: str, seed: int = 42):
        """Generate complete episode from script"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        script = self.parse_script(script_path)
        segments = []

        for section in script.sections:
            # Generate audio for section
            audio = self.generate_section(
                text=section.text,
                anchor_idx=section.anchor_idx,
                prosody_hints=section.prosody
            )
            segments.append(audio)

        # Concatenate with crossfade
        episode = self.concatenate_segments(segments)

        # Post-process
        episode = self.postprocess(episode)

        return episode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = EpisodeGenerator(args.model)
    audio = generator.generate_episode(args.script, args.seed)

    # Save output
    save_audio(audio, args.output, sample_rate=24000)
    print(f"Episode saved to {args.output}")

if __name__ == "__main__":
    main()
```

### A.3 Multi-Speaker Orchestration

```python
class DialogueOrchestrator:
    """Orchestrate multi-speaker dialogue generation"""

    def __init__(self, speaker_configs: dict):
        self.speakers = self.load_speakers(speaker_configs)
        self.timing_model = TimingModel()
        self.mixer = AudioMixer()

    def render_conversation(self, dialogue: list) -> np.ndarray:
        """Render natural conversation with proper timing"""

        # Calculate timing for each utterance
        timed_lines = self.timing_model.calculate_timing(dialogue)

        # Generate audio for each speaker
        tracks = {}
        for line in timed_lines:
            if line.speaker not in tracks:
                tracks[line.speaker] = []

            audio = self.speakers[line.speaker].generate(
                text=line.text,
                emotion=line.emotion,
                pace=line.pace
            )

            tracks[line.speaker].append({
                "audio": audio,
                "start": line.start_time,
                "end": line.end_time,
            })

        # Mix all tracks with ducking
        mixed = self.mixer.mix_dialogue(
            tracks=tracks,
            duck_amount=-6.0,
            crossfade_ms=50
        )

        return mixed

    def add_spatial_positioning(
        self,
        audio: np.ndarray,
        speaker_positions: dict
    ) -> np.ndarray:
        """Create stereo scene with speaker positioning"""

        stereo = np.zeros((len(audio), 2))

        for speaker, position in speaker_positions.items():
            # position: -1.0 (left) to 1.0 (right)
            left_gain = (1.0 - position) / 2.0
            right_gain = (1.0 + position) / 2.0

            track = self.get_speaker_track(audio, speaker)
            stereo[:, 0] += track * left_gain
            stereo[:, 1] += track * right_gain

        return stereo
```

---

## Appendix B: Development Environment

### B.1 Directory Structure

```shell
voice-podcast-template/
├── voice-dataset-kit/          # ✅ Complete
│   ├── tools/
│   │   ├── audio/
│   │   ├── subtitles/
│   │   └── util/
│   └── cli/
│       ├── build_dataset.py
│       └── reviewer.py
│
├── model-training/             # To be created
│   ├── src/
│   │   ├── data/
│   │   ├── models/
│   │   ├── training/
│   │   └── evaluation/
│   ├── configs/
│   │   ├── lora_default.yaml
│   │   └── evaluation.yaml
│   └── scripts/
│       ├── train.py
│       └── evaluate.py
│
├── podcast-synthesis/          # To be created
│   ├── src/
│   │   ├── inference/
│   │   ├── orchestration/
│   │   ├── postprocess/
│   │   └── utils/
│   ├── configs/
│   │   ├── synthesis.yaml
│   │   └── mixing.yaml
│   └── scripts/
│       ├── synthesize.py
│       └── synthesize_multi.py
│
├── data/                       # Dataset storage
│   ├── train/
│   ├── val/
│   ├── test/
│   └── ref/
│
├── models/                     # Trained models
│   └── voice_v1/
│       ├── best.pt
│       ├── config.yaml
│       └── anchors/
│
└── episodes/                   # Generated content
    ├── scripts/
    └── output/
```

### B.2 Development Workflow

```bash
# 1. Prepare dataset
uv run voicekit-build --in-dir recordings/ --out-dir data/

# 2. Review and fix transcripts
uv run voicekit-review --out-dir data/

# 3. Train model
uv run python model-training/scripts/train.py \
    --dataset data/ \
    --config model-training/configs/lora_default.yaml \
    --output models/voice_v1/

# 4. Evaluate model
uv run python model-training/scripts/evaluate.py \
    --model models/voice_v1/best.pt \
    --test-set data/test/

# 5. Generate episode
uv run python podcast-synthesis/scripts/synthesize.py \
    --model models/voice_v1/best.pt \
    --script episodes/scripts/ep001.md \
    --output episodes/output/ep001.wav

# 6. Generate multi-speaker episode
uv run python podcast-synthesis/scripts/synthesize_multi.py \
    --speakers configs/speakers.yaml \
    --script episodes/scripts/dialogue.json \
    --output episodes/output/multi.wav
```

---

## Appendix C: Monitoring & Observability

### C.1 Training Metrics

```python
# Logged to TensorBoard
metrics = {
    "train/loss": 0.123,
    "train/learning_rate": 5e-5,
    "val/wer": 0.065,
    "val/speaker_similarity": 0.82,
    "val/mos_proxy": 4.1,
    "system/gpu_memory_gb": 18.5,
    "system/batch_time_ms": 1250,
}
```

### C.2 Inference Metrics

```json
{
  "inference_metrics": {
    "utterances_generated": 145,
    "total_duration_sec": 1823.4,
    "generation_time_sec": 892.1,
    "realtime_factor": 2.04,
    "peak_memory_gb": 12.3,
    "cache_hits": 89,
    "errors": []
  }
}
```

### C.3 Quality Tracking

```sql
-- Example metrics database schema
CREATE TABLE generation_quality (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50),
    episode_id VARCHAR(100),
    wer FLOAT,
    speaker_similarity FLOAT,
    mos_score FLOAT,
    user_rating INTEGER,
    feedback TEXT
);
```

---

This Technical Design Document provides a comprehensive blueprint for implementing the voice cloning and podcast synthesis pipeline, with clear phases, technical specifications, and practical code examples to guide development.

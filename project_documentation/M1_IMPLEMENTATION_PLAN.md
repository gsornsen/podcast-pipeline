# M1 Implementation Plan: Single-Speaker Training & Synthesis

## 🎯 Overview

This document outlines the implementation plan for **M1: Single-Speaker Training & Synthesis**, building on the completed M0 data preparation infrastructure. M1 will deliver a fully functional voice cloning system capable of training on 30-90 minutes of curated audio and generating natural-sounding synthetic speech.

## 📊 Current Status

### ✅ Infrastructure Complete

- **MCP Servers**: taskqueue, temporal-mcp, RedisMCPServer, hugging-face, jam, playwright, github
- **Agent Coordination**: 30+ specialized Claude Code agents ready for deployment
- **Development Environment**: uv, ruff, mypy, Python 3.13+ configured
- **Version Control**: Git repository with proper branch structure

### ✅ M0 Foundation Complete

- **voice-dataset-kit**: Fully implemented segmentation, QC, and review tools
- **Data Structure**: train/val/test directories with metadata.jsonl format
- **Audio Processing**: 24kHz, 16-bit PCM, mono pipeline established
- **Quality Control**: Hum detection, plosive filtering, speaker verification

### 🎯 Ready for M1

- **Next Step**: Audio curation and data population
- **Target**: 30-90 minutes of high-quality training data
- **Goal**: Functional single-speaker voice cloning system

## 📋 Implementation Phases

### Phase 1: Data Curation & Population (1-2 days)

#### Audio Source Requirements

- **Duration**: 30-90 minutes total (start with 30-60 min)
- **Quality**: Clean recordings, minimal background noise
- **Consistency**: Same microphone/recording chain preferred
- **Content**: Natural speech, varied prosody
- **Format**: Any common audio format (.wav, .mp3, .flac, etc.)

#### Data Preparation Workflow

1. **Source Selection**
   - Identify podcast episodes, interviews, or recordings
   - Ensure content represents target voice characteristics
   - Verify audio quality and consistency

2. **Processing Pipeline**

   ```bash
   # Use existing voice-dataset-kit tools
   uv run python -m voice-dataset-kit.cli.build_dataset \
     --in_dir raw_audio/ \
     --out_dir data/ \
     --sr 24000 \
     --min_sec 5 --max_sec 15 \
     --margin_ms 150 \
     --hum_autonotch --mains 60 \
     --spk_sim_thresh 0.55
   ```

3. **Quality Validation**
   - Review segmented clips with reviewer tool
   - Aim for ≥90% clips between 5-15 seconds
   - Target ≤5% clips with defects
   - Ensure diverse prosodic patterns

#### Success Criteria

- [ ] 30-60 minutes of curated training data
- [ ] ≥90% clips meet duration requirements
- [ ] Quality metrics pass thresholds
- [ ] Train/val/test splits properly populated

### Phase 2: Training Infrastructure Setup (2-3 days)

#### Model Integration

1. **Sesame CSM-1B Setup**
   - Clone/setup Sesame CSM-1B repository
   - Install model dependencies and requirements
   - Verify RTX 4090 compatibility (≤24GB VRAM)
   - Test base model inference

2. **LoRA Configuration**
   - Implement LoRA adaptation layers
   - Configure ranks 8-16 for optimal performance
   - Target audio decoder + selected backbone layers
   - Set up mixed precision training (bf16/fp16)

3. **Training Pipeline Development**

   ```python
   # Key training configuration
   TRAINING_CONFIG = {
       "batch_size": 2,
       "gradient_accumulation_steps": 4,
       "learning_rate": 1e-4,
       "warmup_steps": 100,
       "max_steps": 2000,
       "eval_steps": 100,
       "save_steps": 200,
       "lora_rank": 16,
       "lora_alpha": 32,
       "mixed_precision": "bf16",
       "gradient_checkpointing": True
   }
   ```

#### Infrastructure Components

1. **Data Loading**
   - Custom PyTorch DataLoader for voice-dataset-kit format
   - Efficient audio loading and preprocessing
   - Dynamic batching for variable-length sequences

2. **Training Loop**
   - Mixed precision training with autocast
   - Gradient checkpointing for memory efficiency
   - Learning rate scheduling (warmup + cosine decay)
   - Early stopping on validation metrics

3. **Monitoring & Logging**
   - Integration with Temporal workflows
   - Progress tracking via taskqueue MCP
   - Metrics logging (loss, WER, CER)
   - VRAM usage monitoring

#### Success Criteria

- [ ] Training environment operational on RTX 4090
- [ ] LoRA configuration optimized for memory
- [ ] Data pipeline processes voice-dataset-kit output
- [ ] Monitoring infrastructure functional

### Phase 3: Model Training & Evaluation (1-2 weeks)

#### Training Execution

1. **Initial Training Run**
   - Start with 30-minute dataset for proof-of-concept
   - Monitor training metrics and convergence
   - Validate memory usage stays ≤22GB VRAM
   - Adjust hyperparameters as needed

2. **Iterative Improvement**
   - Experiment with LoRA ranks (8, 12, 16)
   - Tune learning rate and batch size
   - Optimize for target metrics:
     - WER ≤ 7%
     - CER ≤ 3%
     - ECAPA cosine similarity ≥ 0.75

3. **Evaluation Framework**
   - Automated WER/CER evaluation with Whisper
   - Speaker similarity measurement with ECAPA
   - Subjective quality assessment (MOS-lite)
   - Inference speed benchmarking

#### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| WER | ≤ 7% | Whisper transcription vs ground truth |
| CER | ≤ 3% | Character-level accuracy |
| Speaker Similarity | ≥ 0.75 | ECAPA cosine vs reference |
| Inference Speed | ≥ 1.0x realtime | 30min episode ≤ 30min |
| Memory Usage | ≤ 22GB VRAM | Peak during training |
| Model Size | ≤ 500MB | LoRA adapter + config |

#### Success Criteria

- [ ] Training converges within memory constraints
- [ ] All performance targets achieved
- [ ] Model quality validated through evaluation
- [ ] Inference pipeline demonstrates realtime capability

### Phase 4: Inference Pipeline Development (3-5 days)

#### Synthesis API

1. **Core Inference Engine**
   - Text-to-speech generation with LoRA adapter
   - Anchor-based identity stabilization
   - Deterministic generation with seed control
   - Batch processing for efficiency

2. **FastAPI Service**

   ```python
   @app.post("/synthesize")
   async def synthesize_speech(
       text: str,
       speaker_id: str,
       config: SynthesisConfig
   ) -> AudioResponse:
       # Generate speech with trained model
       pass
   ```

3. **Post-Processing Pipeline**
   - Audio normalization to -19 LUFS (mono)
   - True peak limiting ≤ -1 dBTP
   - Optional de-essing and noise reduction
   - Format conversion and export

#### Episode Generation

1. **Script Processing**
   - Markdown/JSON script parsing
   - SSML-lite tag support for pauses/emphasis
   - Paragraph and section segmentation
   - Anchor insertion for identity stability

2. **Audio Assembly**
   - Seamless concatenation of generated segments
   - Cross-fade between sections
   - Loudness consistency throughout
   - Timing and pacing control

3. **Export Formats**
   - WAV output (24kHz, 16-bit, mono)
   - Episode metadata JSON with timestamps
   - Progress tracking and resumability

#### Success Criteria

- [ ] API responds with <200ms latency
- [ ] Episode generation completes within realtime
- [ ] Audio quality meets broadcast standards
- [ ] Deterministic output with seed control

## 🔧 Development Environment

### Required Tools

- **Python**: 3.13+ with uv package manager
- **PyTorch**: 2.8+ with CUDA support
- **Audio**: librosa, soundfile, pyloudnorm
- **ML**: transformers, accelerate, peft
- **API**: FastAPI, uvicorn
- **Quality**: ruff, mypy, pytest

### Hardware Requirements

- **GPU**: RTX 4090 (24GB VRAM)
- **RAM**: 32GB+ system memory recommended
- **Storage**: 50GB+ for models and data
- **CPU**: Multi-core for data processing

### Development Setup

```bash
# Clone and setup
git clone <sesame-csm-1b-repo>
cd podcast-pipeline
uv sync --all-extras

# Verify GPU availability
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Success Metrics & Acceptance Criteria

### Technical Metrics

- **Training Throughput**: ≥100 steps/hour on RTX 4090
- **Memory Efficiency**: Peak VRAM ≤22GB during training
- **Model Quality**: WER ≤7%, CER ≤3%, Speaker similarity ≥0.75
- **Inference Speed**: ≥1.0x realtime generation
- **Storage**: LoRA adapter ≤500MB

### Functional Requirements

- **End-to-End Pipeline**: Text → Natural speech
- **Identity Preservation**: Recognizable as target speaker
- **Quality**: Broadcast-ready audio output
- **Reproducibility**: Deterministic with seed control
- **Usability**: Simple CLI and API interfaces

### User Experience

- **Setup Time**: <30 minutes for new environment
- **Training Time**: ≤4 hours for 30-minute dataset
- **Generation Speed**: Real-time or faster
- **Quality Feedback**: Clear metrics and examples

## 🚀 Agent Deployment Strategy

### Primary Implementation Team

- **ai-engineer**: Model architecture and training loop
- **python-pro**: Core implementation and Pythonic patterns
- **ml-engineer**: ML pipeline and data handling
- **mlops-engineer**: Training infrastructure and monitoring

### Supporting Specialists

- **data-engineer**: Dataset pipeline optimization
- **performance-engineer**: Speed and memory optimization
- **test-automator**: Comprehensive testing framework
- **api-designer**: FastAPI service design

### Quality Assurance

- **code-reviewer**: Code quality and standards
- **architect-reviewer**: System design validation
- **debugger**: Issue resolution and troubleshooting

### Coordination

- **multi-agent-coordinator**: Overall project coordination
- **workflow-orchestrator**: Pipeline execution management
- **performance-monitor**: Metrics tracking and reporting

## 📅 Timeline & Milestones

### Week 1: Foundation

- [ ] Audio data curation and processing
- [ ] Training infrastructure setup
- [ ] Initial model integration

### Week 2-3: Core Development

- [ ] Training pipeline implementation
- [ ] Model training and iteration
- [ ] Evaluation framework development

### Week 4: Integration & Testing

- [ ] Inference pipeline development
- [ ] API service implementation
- [ ] End-to-end testing and validation

### Week 5-6: Optimization & Documentation

- [ ] Performance optimization
- [ ] Quality assurance and testing
- [ ] Documentation and examples

## 🔄 Risk Mitigation

### Technical Risks

- **Memory Constraints**: Use gradient checkpointing, smaller batches
- **Training Instability**: Implement proper learning rate scheduling
- **Quality Issues**: Comprehensive evaluation and human feedback
- **Performance Bottlenecks**: Profile and optimize critical paths

### Data Risks

- **Insufficient Quality**: Strict QC criteria and review process
- **Limited Diversity**: Ensure varied prosodic patterns
- **Overfitting**: Proper validation splits and early stopping

### Infrastructure Risks

- **Hardware Failures**: Regular checkpointing and backup
- **Dependency Issues**: Locked requirements and containers
- **Resource Contention**: Dedicated development environment

## 📋 Checklist for Audio Curation

### Pre-Processing

- [ ] Source audio identified and collected
- [ ] Audio quality verified (clean, consistent)
- [ ] Total duration confirmed (30-90 minutes)
- [ ] File formats compatible with voice-dataset-kit

### Processing

- [ ] Run voice-dataset-kit segmentation
- [ ] Complete quality review with reviewer tool
- [ ] Generate train/val/test splits
- [ ] Validate metadata.jsonl format

### Post-Processing

- [ ] Confirm clip duration distribution
- [ ] Verify quality metrics pass thresholds
- [ ] Test data loading in training pipeline
- [ ] Document any processing notes or issues

---

*This plan will be updated as M1 implementation progresses. All changes should be tracked and communicated through the multi-agent-coordinator.*

# Podcast Pipeline - Voice Cloning & Synthesis

An end-to-end pipeline for voice cloning and podcast synthesis, featuring automated dataset preparation, LoRA-based voice model training, and multi-speaker episode generation.

## 🎯 Project Overview

This project implements a complete voice cloning pipeline consisting of three major phases:

- **✅ M0 - Voice Dataset Kit** (Complete): Automated conversion of long recordings into clean, segmented training data
- **🎯 M1 - Model Training** (Ready): LoRA fine-tuning of Sesame CSM-1B for single-speaker voice cloning
- **📅 M2 - Podcast Synthesis** (Planned): Multi-speaker episode generation with automated mixing

### Current Status

**Phase M0 Complete**: Full data preparation pipeline implemented and tested
**Phase M1 Ready**: Infrastructure and coordination systems operational, awaiting audio curation

## 🚀 Quick Start

### Prerequisites

- **Hardware**: RTX 4090 (24GB VRAM) or equivalent
- **Python**: 3.13+ with uv package manager
- **Audio**: Clean recordings for voice cloning (30-90 minutes recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd podcast-pipeline

# Install with uv (recommended)
uv sync --group dev

# Or with pip
pip install -e .
```

### Basic Usage

```bash
# Process raw audio recordings into training dataset
python -m voice_dataset_kit.cli \
    --input /path/to/recordings \
    --output ./data \
    --target-loudness -19.0 \
    --min-duration 5.0 \
    --max-duration 15.0

# Validate dataset quality
python -m voice_dataset_kit.cli validate ./data
```

## 📁 Project Structure

```
podcast-pipeline/
├── voice-dataset-kit/          # M0: Data preparation pipeline
│   ├── cli/                    # Command-line interfaces
│   ├── core/                   # Audio processing engines
│   ├── quality/                # Data validation tools
│   └── review/                 # Dataset review interfaces
├── training/                   # M1: Model training (planned)
├── synthesis/                  # M2: Podcast generation (planned)
├── tests/                      # Comprehensive test suite
└── docs/                       # Technical documentation
```

## 🎵 Audio Standards

All audio processing maintains strict standards:

- **Sample Rate**: 24,000 Hz (never resampled)
- **Bit Depth**: 16-bit PCM
- **Channels**: Mono (single channel)
- **Loudness**: -19.0 LUFS target
- **Peak**: -1.0 dBTP maximum

## 🔧 Technical Specifications

### Voice Dataset Kit (M0 - Complete)

**Automated Audio Processing**:
- WebRTC VAD-based segmentation with smart merging
- Pyloudnorm loudness normalization to broadcast standards
- Automatic train/validation/test splits with source grouping
- Quality validation with comprehensive metrics

**Dataset Generation**:
- Metadata CSV generation with pipe-separated format
- Reference clip extraction for inference validation
- Configurable duration and quality thresholds
- Support for multiple input formats (WAV, FLAC, MP3)

**Quality Assurance**:
- Real-time processing validation
- Statistical analysis of segment distributions
- Audio quality metrics and reporting
- Manual review interface for transcript correction

### Model Training Infrastructure (M1 - Ready)

**Target Model**: Sesame CSM-1B with LoRA adaptation
**Training Framework**: PyTorch 2.8+ with mixed precision (bf16)
**Memory Optimization**: Gradient checkpointing for 24GB VRAM limit
**Evaluation Metrics**: WER ≤7%, CER ≤3%, ECAPA cosine ≥0.75

### Synthesis Pipeline (M2 - Planned)

**Single-Speaker Generation**: FastAPI service with SSML support
**Multi-Speaker Orchestration**: Cross-model dialogue coordination
**Audio Post-Processing**: Professional mixing and mastering
**Episode Assembly**: Automated podcast episode generation

## 🤖 Agent Coordination System

This project employs a sophisticated multi-agent coordination system with 30+ specialized Claude Code agents:

### Meta-Orchestration Team
- **multi-agent-coordinator**: Overall project coordination
- **workflow-orchestrator**: Pipeline execution management
- **task-distributor**: Intelligent work allocation
- **context-manager**: Project state consistency
- **performance-monitor**: System metrics tracking

### Development Specialists
- **AI/ML Team**: ai-engineer, ml-engineer, data-engineer, nlp-engineer
- **Infrastructure**: devops-engineer, platform-engineer, dependency-manager
- **Quality Assurance**: code-reviewer, test-automator, performance-engineer
- **Documentation**: technical-writer, documentation-engineer

## 📊 Performance Targets

### Data Processing (M0)
- **Throughput**: Process 60+ minutes of audio in <5 minutes
- **Quality**: ≥90% of segments between 5-15 seconds
- **Accuracy**: Manual transcript review achieves >99% accuracy

### Model Training (M1)
- **Training Time**: <24 hours on RTX 4090 for 30-60 minutes of data
- **Memory Usage**: <22GB VRAM peak with gradient checkpointing
- **Quality Gates**: WER ≤7%, identity similarity ≥0.75

### Synthesis (M2)
- **Real-time Factor**: 30-minute episode generated in ≤30 minutes
- **Quality**: Broadcast-ready audio with professional mixing
- **Scalability**: Support for 2-8 speakers per episode

## 🔐 Privacy & Security

- **Local Processing**: All audio processing occurs locally, no cloud uploads
- **Consent Management**: Built-in voice usage permission tracking
- **Model Distribution**: Only LoRA adapters shared (<500MB)
- **Data Protection**: Audio content never logged or transmitted

## 🏗️ Development Workflow

### Code Quality Standards
- **Linting**: Ruff with strict settings
- **Type Checking**: MyPy with comprehensive type coverage
- **Testing**: Automated test suite with >90% coverage
- **Review**: Multi-agent code review process

### Collaboration Patterns
- **Pair Programming**: AI specialists work in coordinated pairs
- **Review Chains**: Sequential reviews for critical components
- **Agent Coordination**: Task distribution based on specialist capabilities

## 📈 Roadmap

### Immediate Next Steps (1-2 days)
1. **Audio Curation**: Collect and process target speaker recordings
2. **Dataset Validation**: Run complete quality assurance pipeline
3. **Training Preparation**: Verify infrastructure and dependencies

### M1 Implementation (2-4 weeks)
1. **Sesame Integration**: Model loading and LoRA configuration
2. **Training Pipeline**: Distributed training with monitoring
3. **Evaluation Framework**: Comprehensive model assessment
4. **Inference Service**: FastAPI endpoint for synthesis

### M2 Planning (Future)
1. **Multi-Speaker Support**: Cross-model coordination
2. **Dialogue Timing**: Natural conversation flow
3. **Audio Mixing**: Professional episode post-production

## 🤝 Contributing

This project uses a multi-agent development approach. Contributors should:

1. Review the agent coordination documentation in `CLAUDE.md`
2. Follow established code standards and patterns
3. Coordinate through the appropriate specialist agents
4. Submit work through the review chain process

## 📄 License

[License information to be added]

## 🔗 Links

- **Documentation**: Full technical documentation in `/docs`
- **API Reference**: Auto-generated API docs (coming in M1)
- **Examples**: Sample workflows and configurations
- **Issues**: Project tracking and coordination

---

**Current Phase**: M0 Complete, M1 Ready for Audio Curation
**Last Updated**: 2025-01-28
**Maintained By**: multi-agent-coordinator with documentation-engineer
# CLAUDE.md - Voice Cloning & Podcast Synthesis Pipeline

## 🎯 Project Overview

This project implements an end-to-end pipeline for voice cloning and podcast synthesis, consisting of three major components:

1. **Voice Dataset Kit** (✅ Complete) - Converts long recordings into clean, segmented utterances
2. **Model Training** (🎯 M1 - Ready to Start) - LoRA fine-tuning of Sesame CSM-1B for voice cloning
3. **Podcast Synthesis** (📅 M2 - Planned) - Multi-speaker episode generation with mixing

### Current Status

- **M0**: Data preparation toolchain - ✅ COMPLETE
- **M1**: Single-speaker training & synthesis - 🎯 READY TO START (awaiting audio curation)
- **M2**: Multi-speaker orchestration - 📅 PLANNED (4-6 weeks after M1)

### Technical Constraints

- **Hardware**: Single RTX 4090 (24GB VRAM)
- **Audio Format**: 24kHz, 16-bit PCM, mono throughout
- **Model**: Sesame CSM-1B with LoRA adaptation
- **Framework**: PyTorch 2.8+ with mixed precision (bf16)

## 🤖 Agent Team Composition

### Meta-Orchestration Team (Priority 1)

All agents from `09-meta-orchestration/` work together to coordinate the entire project:

- **multi-agent-coordinator**: Overall project coordination and agent synchronization
- **workflow-orchestrator**: Pipeline design and execution flow management
- **task-distributor**: Intelligent work allocation based on agent capabilities
- **context-manager**: Maintains project state and information consistency
- **agent-organizer**: Assembles optimal agent teams for specific tasks
- **performance-monitor**: Tracks system metrics and training performance
- **error-coordinator**: Handles failures and recovery strategies
- **knowledge-synthesizer**: Extracts learnings and best practices

### Core Development Team

#### AI/ML Specialists

- **python-pro**: Primary implementation language, ensures Pythonic patterns
- **ai-engineer**: Model architecture, training loop implementation
- **ml-engineer**: ML pipeline, data loaders, evaluation metrics
- **mlops-engineer**: Training infrastructure, experiment tracking
- **data-engineer**: Dataset pipeline, data quality, splits management
- **data-scientist**: Statistical analysis, model evaluation
- **prompt-engineer**: Optimizes prompts for TTS generation
- **nlp-engineer**: Text processing, tokenization, SSML parsing

#### Infrastructure & DevOps

- **devops-engineer**: CI/CD, containerization, local development, ArgoCD
- **platform-engineer**: Development environment, tooling, home-k8s integration
- **dependency-manager**: Package management with uv, version control

### Quality Assurance Team

- **code-reviewer**: Reviews all code for quality and standards
- **test-automator**: Builds comprehensive test suites
- **performance-engineer**: Optimizes training and inference speed
- **debugger**: Resolves complex issues
- **architect-reviewer**: Validates system design decisions

### Documentation & Support

- **technical-writer**: API documentation, user guides
- **documentation-engineer**: Maintains technical documentation
- **cli-developer**: Creates user-friendly CLI interfaces

## 🔄 Collaboration Workflows

### 1. Feature Development Workflow

```shell
1. workflow-orchestrator creates execution plan
2. task-distributor assigns work to specialists
3. python-pro + ai-engineer implement core features
4. code-reviewer validates implementation
5. test-automator creates tests
6. performance-engineer optimizes if needed
7. knowledge-synthesizer documents learnings
```

### 2. Model Training Pipeline

```shell
1. data-engineer prepares dataset
2. ml-engineer implements training loop
3. mlops-engineer sets up infrastructure
4. ai-engineer tunes hyperparameters
5. performance-monitor tracks metrics
6. error-coordinator handles failures
7. data-scientist evaluates results
```

### 3. Multi-Agent Pair Programming

For complex tasks, agents work in pairs:

- **python-pro + ai-engineer**: Core model implementation
- **ml-engineer + mlops-engineer**: Training infrastructure
- **data-engineer + data-scientist**: Dataset quality
- **test-automator + debugger**: Testing and debugging

### 4. Review Chains

Critical code paths require sequential reviews:

1. Initial implementation by specialist
2. code-reviewer for standards
3. architect-reviewer for design
4. performance-engineer for optimization
5. security-engineer for safety (if applicable)

## 📐 Code Standards & Patterns

### Audio Processing Standards

```python
AUDIO_CONFIG = {
    "sample_rate": 24000,      # Hz - NEVER CHANGE
    "bit_depth": 16,           # bits - REQUIRED
    "channels": 1,             # mono - ENFORCED
    "format": "PCM_16",        # Linear PCM
    "loudness_target": -19.0,  # LUFS (mono)
    "true_peak": -1.0,         # dBTP
}
```

### PyTorch Model Patterns

```python
# Always use mixed precision for training
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(inputs)

# Gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Deterministic generation
torch.manual_seed(seed)
np.random.seed(seed)
```

### CLI Design Pattern

All CLIs follow this structure:

```python
# Modern CLI with clear help text
@click.command()
@click.option('--input', required=True, help='Input directory')
@click.option('--output', required=True, help='Output directory')
@click.option('--config', type=click.Path(exists=True))
def command(input, output, config):
    """Clear description of what this command does."""
    pass
```

### Error Handling

```python
# Consistent error handling pattern
class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass

class DataError(PipelineError):
    """Data-related errors"""
    pass

# Always provide context
try:
    process_audio(file)
except Exception as e:
    logger.error(f"Failed processing {file}: {e}")
    raise DataError(f"Audio processing failed: {file}") from e
```

### Logging Standards

```python
import logging

# Consistent logger naming
logger = logging.getLogger(__name__)

# Structured logging
logger.info("Processing segment", extra={
    "file": filename,
    "duration": duration_sec,
    "sample_rate": sr
})
```

## 🔍 Quality Gates

### Code Quality Checks

1. **Linting & Formatting**: `ruff check` and `ruff format` must pass
2. **Type Checking**: `mypy` with strict settings
3. **Security**: `bandit` security scan must pass
4. **Pre-commit**: All hooks must pass

### Model Training Gates

1. **Data Quality**: ≥90% clips between 5-15s
2. **Training Metrics**: WER ≤7%, CER ≤3%
3. **Identity Similarity**: ECAPA cosine ≥0.75
4. **Performance**: 30min episode ≤30min generation

### Review Requirements

- All code requires review by code-reviewer
- ML changes require ai-engineer approval
- Infrastructure changes require devops-engineer approval
- API changes require api-designer review

## 🎯 Implementation Priorities

### Current Status: Ready for M1 Implementation 🎯

**Infrastructure Complete**: MCP servers operational, agent coordination established
**M0 Foundation**: voice-dataset-kit fully implemented and tested
**Next Step**: Audio curation to populate training data

### Phase 1: Audio Curation & Data Population (Next - 1-2 days)

**Lead**: data-engineer, python-pro
**Support**: performance-engineer
**Focus**:

- Source 30-90 minutes of target speaker audio
- Run voice-dataset-kit processing pipeline
- Quality validation and review

### Phase 2: Model Training Infrastructure (1-2 weeks)

**Lead**: ai-engineer, ml-engineer
**Support**: python-pro, mlops-engineer
**Focus**:

- Sesame CSM-1B integration and LoRA setup
- Training pipeline with mixed precision
- Evaluation framework and monitoring

### Phase 3: Single-Speaker Synthesis (2-3 weeks)

**Lead**: ai-engineer, nlp-engineer
**Support**: python-pro, prompt-engineer
**Focus**:

- Inference pipeline and FastAPI service
- Script parsing and SSML support
- Audio post-processing and episode assembly

### Phase 4: Multi-Speaker Orchestration (M2 - Future)

**Lead**: ai-engineer, data-engineer
**Support**: performance-engineer
**Focus**:

- Multi-model management
- Dialogue timing
- Audio mixing

## 🚀 Agent Activation Patterns

### For Training Tasks

```shell
Activate: [ai-engineer, ml-engineer, mlops-engineer, data-scientist]
Coordinate: multi-agent-coordinator
Monitor: performance-monitor
```

### For Data Processing

```shell
Activate: [data-engineer, python-pro, performance-engineer]
Coordinate: workflow-orchestrator
Monitor: error-coordinator
```

### For Testing

```shell
Activate: [test-automator, qa-expert, debugger]
Coordinate: task-distributor
Monitor: performance-monitor
```

## 📊 Success Metrics Tracking

The performance-monitor agent tracks:

- Training loss convergence
- Inference speed (realtime factor)
- Memory usage (VRAM peak)
- Dataset processing throughput
- Model quality metrics (WER, MOS, similarity)

## 🔐 Security & Privacy

- **Local Processing Only**: No cloud uploads of voice data
- **Consent Management**: Track voice usage permissions
- **Model Distribution**: LoRA adapters only (<500MB)
- **No Logging of Audio**: Only metadata is logged

## 📝 Knowledge Management

The knowledge-synthesizer maintains:

- Lessons learned after each sprint
- Best practices discovered
- Common pitfalls and solutions
- Performance optimization techniques
- Successful hyperparameter configurations

## 🔄 Continuous Improvement

After each milestone:

1. knowledge-synthesizer extracts learnings
2. context-manager updates project knowledge
3. agent-organizer refines team composition
4. workflow-orchestrator optimizes processes

## 🎓 Agent Learning Points

### What Works Well

- Gradient checkpointing essential for 4090 memory limits
- Mixed precision bf16 provides best stability
- LoRA rank 16 optimal for voice adaptation
- 30-60 minutes of data sufficient for good quality

### Common Pitfalls to Avoid

- Don't skip data QC - bad data ruins models
- Always use deterministic seeds for reproducibility
- Monitor VRAM usage closely during training
- Validate audio format consistency throughout

## 🚦 Quick Start for New Agents

1. Read this CLAUDE.md thoroughly
2. Check current milestone status in PRD.md
3. Review TDD.md for technical specifications
4. Coordinate with multi-agent-coordinator
5. Follow established patterns and standards
6. Submit work through review chain

## 📞 Communication Protocols

- **Status Updates**: Report to multi-agent-coordinator
- **Technical Questions**: Consult relevant specialist agent
- **Blockers**: Escalate to error-coordinator
- **Design Decisions**: Review with architect-reviewer
- **Performance Issues**: Alert performance-monitor

## 🎯 Current Project Status & Next Steps

### ✅ Completed Infrastructure
- **Agent Coordination**: 30+ Claude Code subagents configured and ready
- **MCP Servers**: taskqueue, temporal-mcp, RedisMCPServer, hugging-face, jam, playwright, github
- **Development Environment**: uv, ruff, mypy, Python 3.13+ toolchain established
- **M0 Data Pipeline**: voice-dataset-kit fully implemented with segmentation, QC, and review tools

### 🎯 Immediate Next Step: Audio Curation
- **Goal**: Collect 30-90 minutes of target speaker audio
- **Quality**: Clean recordings, consistent mic/chain preferred
- **Process**: Use existing voice-dataset-kit tools for segmentation and QC
- **Timeline**: 1-2 days to curate and process

### 📋 Ready for M1 Implementation
- **Infrastructure**: All tools and coordination systems operational
- **Foundation**: Complete data processing pipeline tested and validated
- **Team**: Agent specialists assigned and workflow documented
- **Documentation**: Comprehensive implementation plan in `project_documentation/M1_IMPLEMENTATION_PLAN.md`

### 🚀 After Audio Curation
1. **Training Infrastructure**: Sesame CSM-1B integration with LoRA adaptation
2. **Model Training**: Mixed precision training on RTX 4090
3. **Inference Pipeline**: FastAPI service for speech synthesis
4. **Episode Generation**: Full podcast synthesis capability

---

*This document is maintained by the knowledge-synthesizer and updated after each milestone completion. Last updated: Ready for M1 audio curation phase.*
- always use uv run python to run python commands/scripts
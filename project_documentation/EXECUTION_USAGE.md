# EXECUTION_USAGE.md - Orchestration System Usage Guide

## 🎯 Overview

This document provides comprehensive instructions for using the voice cloning podcast pipeline orchestration system. It covers everything from basic system activation to advanced multi-agent workflows, enabling you to efficiently leverage the 30+ specialized Claude Code agents and MCP servers we've configured.

## 📋 Quick Reference

### System Status

- **Project Phase**: M1 - Ready to Start (awaiting audio curation)
- **Infrastructure**: ✅ Complete (MCP servers operational)
- **Agents**: 30+ specialized agents across 11 categories
- **MCP Servers**: taskqueue, temporal-mcp, RedisMCPServer, hugging-face, jam, playwright, github

### Next Steps

1. **Audio Curation**: Collect 30-90 minutes of target speaker audio
2. **M1 Implementation**: Training infrastructure and model development
3. **Synthesis Pipeline**: Inference and episode generation

---

## 🚀 System Activation & Startup

### Prerequisites Check

```bash
# Verify core dependencies
uv --version                    # Python package manager
python --version               # Should be 3.13+
torch --version                # Should be 2.8+
nvidia-smi                     # Verify RTX 4090 available

# Check git status
git status                     # Verify clean working directory
git branch                     # Should be on main or feature branch
```

### MCP Server Startup

Start the required MCP servers for full orchestration capability:

```bash
# 1. Start Redis (for caching, messaging, coordination)
redis-server --daemonize yes --port 6379

# 2. Start Temporal (for workflow orchestration)
temporal server start-dev --ui-port 8080 --db-filename temporal.db

# 3. Verify servers are running
redis-cli ping                 # Should return "PONG"
curl http://localhost:8233     # Temporal UI should respond (may be different port)

# 4. Check MCP server status in Claude Code
# The following servers should be available:
# - taskqueue (✅ Ready)
# - temporal-mcp (✅ Ready)
# - RedisMCPServer (✅ Ready)
# - hugging-face (✅ Ready)
# - jam (✅ Ready)
# - playwright (✅ Ready)
# - github (✅ Ready)
```

### System Health Check

```bash
# Verify Python environment
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import redis; r = redis.Redis(); print(f'Redis: {r.ping()}')"

# Test voice-dataset-kit
uv run python -m voice-dataset-kit.cli.build_dataset --help

# Check project structure
ls -la data/                   # Should show train/val/test dirs
ls -la voice-dataset-kit/      # Should show complete toolkit
```

---

## 🤖 Agent Orchestration Fundamentals

### Meta-Orchestration Layer

The system uses a hierarchical approach with meta-orchestration agents coordinating specialist agents:

#### Primary Coordinators

- **multi-agent-coordinator**: Overall project coordination and agent synchronization
- **workflow-orchestrator**: Pipeline design and execution flow management
- **task-distributor**: Intelligent work allocation based on agent capabilities
- **context-manager**: Maintains project state and information consistency

#### How to Activate Orchestration

**Basic Pattern:**

```shell
@multi-agent-coordinator: Please coordinate [task description] using appropriate specialist agents.
```

**Examples:**

```shell
@multi-agent-coordinator: Please coordinate the implementation of LoRA training pipeline for Sesame CSM-1B using appropriate ML specialists.

@workflow-orchestrator: Design and execute a workflow for processing audio data through voice-dataset-kit and preparing for model training.

@task-distributor: Allocate work for increasing test coverage to 90% across the codebase using appropriate QA specialists.
```

### Agent Categories & Activation

#### AI/ML Specialists

```bash
# For model development and training
@ai-engineer: Implement LoRA adaptation for Sesame CSM-1B
@ml-engineer: Create training data pipeline for voice-dataset-kit format
@mlops-engineer: Set up training infrastructure with monitoring
@data-engineer: Optimize dataset processing and validation
@data-scientist: Analyze model performance and evaluation metrics
```

#### Infrastructure & DevOps

```bash
# For development environment and deployment
@devops-engineer: Create Docker development environment
@platform-engineer: Set up integration with home-k8s infrastructure
@dependency-manager: Update Python dependencies and resolve conflicts
```

#### Quality Assurance

```bash
# For testing and code quality
@code-reviewer: Review training pipeline implementation for quality and standards
@test-automator: Create comprehensive test suite for voice processing
@performance-engineer: Optimize training pipeline for RTX 4090 constraints
@debugger: Investigate and resolve training convergence issues
```

### Multi-Agent Coordination Patterns

#### Pair Programming

```bash
# Combine specialists for complex tasks
@multi-agent-coordinator: Please coordinate @python-pro and @ai-engineer to implement the core training loop with proper Pythonic patterns and ML best practices.

@task-distributor: Assign @ml-engineer and @mlops-engineer to work together on training infrastructure with monitoring integration.
```

#### Sequential Workflows

```bash
# Chain agents for multi-step processes
@workflow-orchestrator: Execute the following sequence:
1. @data-engineer: Process raw audio through voice-dataset-kit
2. @ml-engineer: Create data loaders for training
3. @ai-engineer: Implement LoRA training
4. @code-reviewer: Review all implementation
5. @test-automator: Create tests for the pipeline
```

#### Review Chains

```bash
# Multi-level quality assurance
@multi-agent-coordinator: Please coordinate a review chain for the training pipeline:
1. @code-reviewer: Standards and quality
2. @architect-reviewer: System design validation
3. @performance-engineer: Memory and speed optimization
4. @security-engineer: Security best practices (if applicable)
```

---

## 🔧 MCP Server Integration & Usage

### TaskQueue Management

**Create and Track Work Items:**

```bash
# Create a project for M1 implementation
@multi-agent-coordinator: Please use the taskqueue MCP to create a project for "M1 Single-Speaker Training" with tasks for:
- Audio data curation and processing
- Sesame CSM-1B integration
- LoRA training implementation
- Inference pipeline development
- Testing and validation
```

**Task Status Management:**

```bash
# Update task status
@task-distributor: Please mark "Audio curation" task as in_progress and assign to @data-engineer

# Request task approval
@multi-agent-coordinator: Please update "LoRA implementation" task as completed with details and request approval

# Get project status
@performance-monitor: Please provide current status of M1 project tasks using taskqueue
```

### Temporal Workflow Orchestration

**Complex Multi-Step Workflows:**

```bash
# Create training workflow
@workflow-orchestrator: Please design a Temporal workflow for model training that includes:
- Data validation and preprocessing
- Model initialization and LoRA setup
- Training loop with checkpointing
- Evaluation and metrics collection
- Model validation and export

# Monitor workflow execution
@performance-monitor: Please track the training workflow progress and report on metrics
```

**State Management:**

```bash
# Persistent workflow state
@workflow-orchestrator: Please create a resumable workflow for dataset processing that can recover from interruptions

# Workflow recovery
@error-coordinator: Please resume the training workflow from the last successful checkpoint
```

### Redis Integration

**Caching and Coordination:**

```bash
# Cache expensive computations
@performance-engineer: Please implement Redis caching for audio preprocessing results

# Inter-agent communication
@context-manager: Please use Redis to coordinate state between data processing and training agents

# Progress tracking
@performance-monitor: Please store training metrics in Redis for real-time monitoring
```

### Hugging Face Integration

**Model and Dataset Management:**

```bash
# Search for models
@ai-engineer: Please use hugging-face MCP to search for Sesame CSM-1B compatible models and adapters

# Download dependencies
@ml-engineer: Please use hugging-face tools to download required tokenizers and preprocessing components

# Model evaluation
@data-scientist: Please use hugging-face tools to compare our model performance against similar TTS models
```

### GitHub Integration

**Version Control and Collaboration:**

```bash
# Create feature branches
@git-workflow-manager: Please use github MCP to create a feature branch for M1 implementation

# Pull request management
@code-reviewer: Please use github MCP to create a PR for the training pipeline with comprehensive review

# Issue tracking
@error-coordinator: Please use github MCP to create issues for any bugs or improvements found during testing
```

### Playwright Integration

**UI Testing and Automation:**

```bash
# Test web interfaces
@test-automator: Please use playwright to test any web dashboards or UIs we create for monitoring

# Automated browser tasks
@data-engineer: Please use playwright to automate any web-based data collection if needed
```

### JAM Integration

**Bug Reporting and Analysis:**

```bash
# Analyze bug reports
@debugger: Please use jam MCP to analyze any bug reports from training runs

# Quality assurance
@qa-expert: Please use jam integration to track and categorize any issues found during testing
```

---

## 📊 Task Management & Project Control

### Requesting Work & Validation

#### Standard Work Requests

```bash
# Development tasks
@multi-agent-coordinator: Please coordinate implementation of mixed precision training for the LoRA pipeline

# Quality improvements
@task-distributor: Please assign specialists to increase test coverage to 90% across all modules

# Documentation
@technical-writer: Please create comprehensive API documentation for the training pipeline

# Infrastructure
@devops-engineer: Please create Docker Compose configuration for local development environment
```

#### Validation & Review Requests

```bash
# Code quality validation
@code-reviewer: Please review the training loop implementation for:
- Code quality and standards compliance
- Error handling and edge cases
- Performance optimization opportunities
- Security best practices

# Architecture validation
@architect-reviewer: Please validate the overall system design for:
- Scalability and maintainability
- Component separation and interfaces
- Data flow and error handling
- Integration with existing infrastructure

# Performance validation
@performance-engineer: Please validate that the training pipeline:
- Stays within RTX 4090 memory constraints (≤22GB VRAM)
- Achieves target training throughput (≥100 steps/hour)
- Demonstrates real-time inference capability
```

#### Re-work Requests

```bash
# Request modifications
@multi-agent-coordinator: Please coordinate re-work of the data loading pipeline to:
- Improve memory efficiency
- Add better error handling
- Support dynamic batch sizing
- Include progress monitoring

# Iterative improvement
@workflow-orchestrator: Please execute an iterative improvement cycle:
1. @performance-engineer: Profile current training performance
2. @ai-engineer: Implement identified optimizations
3. @test-automator: Validate improvements with benchmarks
4. @performance-monitor: Compare before/after metrics
```

### Priority Management

#### Priority Levels

```bash
# P0 - Critical/Blocking
@error-coordinator: URGENT - Training is failing due to CUDA memory errors, please coordinate immediate resolution

# P1 - High Priority
@task-distributor: Please prioritize implementation of evaluation metrics as they're needed for model validation

# P2 - Normal Priority
@multi-agent-coordinator: Please schedule documentation updates when development capacity is available

# P3 - Low Priority/Nice to Have
@platform-engineer: Please consider adding Grafana dashboard integration when other priorities are complete
```

#### Dependency Management

```bash
# Sequential dependencies
@workflow-orchestrator: Please ensure audio curation is completed before starting training implementation

# Parallel work coordination
@task-distributor: Please coordinate parallel work on:
- @data-engineer: Dataset processing optimization
- @ai-engineer: Model architecture implementation
- @devops-engineer: Infrastructure setup
While ensuring they don't conflict

# Blocking issue resolution
@error-coordinator: Please resolve the LoRA configuration issue blocking training before proceeding with evaluation
```

---

## 🛠️ Common Development Workflows

### Code Quality Improvements

#### Increase Test Coverage

```bash
@test-automator: Please increase test coverage to 90% by:
1. Analyzing current coverage gaps with pytest-cov
2. Creating unit tests for untested functions
3. Adding integration tests for key workflows
4. Implementing property-based tests for data processing
5. Setting up coverage reporting and CI integration

# Follow-up validation
@qa-expert: Please validate that the new tests effectively catch regressions and edge cases
```

#### Code Review & Quality

```bash
@code-reviewer: Please perform comprehensive code review focusing on:
- Pythonic patterns and idioms
- Type safety and mypy compliance
- Error handling and edge cases
- Performance optimization opportunities
- Security best practices
- Documentation completeness

# Quality metrics
@performance-monitor: Please track code quality metrics:
- Cyclomatic complexity
- Technical debt indicators
- Type coverage percentage
- Documentation coverage
```

#### Refactoring & Modernization

```bash
@refactoring-specialist: Please refactor the audio processing module to:
- Improve code organization and modularity
- Reduce complexity and duplication
- Enhance type safety
- Optimize performance
- Maintain backward compatibility

@legacy-modernizer: Please modernize any outdated patterns to current Python 3.13+ standards
```

### Infrastructure & DevOps

#### Docker & Containerization

```bash
@devops-engineer: Please create Docker infrastructure:
1. Multi-stage Dockerfile for development and production
2. Docker Compose for local development with all services
3. Include Redis, Temporal, and monitoring tools
4. Optimize for RTX 4090 GPU access
5. Include development tools (jupyter, monitoring)

# Compose file example request
@platform-engineer: Please create docker-compose.yml with services:
- app: Main application with GPU access
- redis: Caching and coordination
- temporal: Workflow orchestration
- jupyter: Development notebooks
- monitoring: Prometheus/Grafana stack
```

#### CI/CD Pipeline

```bash
@devops-engineer: Please implement CI/CD pipeline with:
1. Automated testing on PR creation
2. Code quality checks (ruff, mypy, bandit)
3. Security scanning
4. Performance benchmarking
5. Automated deployment to staging
6. Integration with home-k8s for production

@deployment-engineer: Please set up deployment automation for model artifacts and training pipelines
```

#### Monitoring & Observability

```bash
@performance-monitor: Please implement comprehensive monitoring:
1. Training metrics dashboard
2. System resource monitoring (GPU, memory, disk)
3. Application performance metrics
4. Error tracking and alerting
5. Integration with home-k8s Prometheus/Grafana

@sre-engineer: Please set up alerting for:
- Training failures or convergence issues
- Resource exhaustion (VRAM, disk space)
- Service availability
- Performance degradation
```

### Documentation & User Experience

#### API Documentation

```bash
@api-documenter: Please create comprehensive API documentation:
1. OpenAPI/Swagger specs for REST endpoints
2. Interactive documentation portal
3. Code examples and tutorials
4. Authentication and rate limiting docs
5. SDK generation for Python clients

@technical-writer: Please create user guides for:
- Getting started with voice cloning
- Training custom models
- API integration examples
- Troubleshooting common issues
```

#### CLI Development

```bash
@cli-developer: Please create user-friendly CLI tools:
1. Modern CLI with click framework
2. Interactive TUI for complex operations
3. Progress bars and status indicators
4. Comprehensive help and documentation
5. Shell completion support

# Example CLI structure
voice-pipeline train --config training.yaml --data data/ --output models/
voice-pipeline synthesize --model models/speaker.lora --text "Hello world" --output audio.wav
voice-pipeline evaluate --model models/speaker.lora --test-data data/test/
```

### Testing & Validation

#### Comprehensive Testing

```bash
@test-automator: Please create comprehensive test suite:
1. Unit tests for all core functions (pytest)
2. Integration tests for end-to-end workflows
3. Performance benchmarks for training and inference
4. Load testing for API endpoints
5. Property-based testing for data processing

@qa-expert: Please implement quality assurance:
1. Manual testing protocols
2. Acceptance criteria validation
3. User experience testing
4. Performance regression testing
5. Security testing procedures
```

#### Model Validation

```bash
@data-scientist: Please implement model validation framework:
1. Automated evaluation metrics (WER, CER, MOS)
2. Speaker similarity measurement (ECAPA)
3. A/B testing framework for model comparison
4. Statistical significance testing
5. Continuous evaluation pipeline

@ml-engineer: Please create model testing infrastructure:
1. Synthetic test data generation
2. Regression testing for model updates
3. Performance benchmarking suite
4. Model artifact validation
5. Deployment readiness checks
```

---

## 🎯 Specific Use Cases & Examples

### Audio Curation & Processing

#### Data Collection

```bash
@data-engineer: Please help me curate audio data by:
1. Analyzing audio quality of provided source files
2. Running voice-dataset-kit segmentation with optimal settings
3. Performing quality validation and review
4. Generating training/validation splits
5. Creating data quality report with recommendations

# Specific command examples
@multi-agent-coordinator: Please coordinate audio processing:
- Use voice-dataset-kit to process ~/audio_sources/
- Target 30-60 minutes of clean segments
- Apply hum filtering and speaker verification
- Generate metadata with quality metrics
- Prepare data for M1 training pipeline
```

#### Quality Assurance

```bash
@qa-expert: Please validate processed audio data:
1. Check that ≥90% of clips are between 5-15 seconds
2. Verify audio quality metrics meet thresholds
3. Ensure consistent audio format (24kHz, 16-bit, mono)
4. Validate transcript accuracy and alignment
5. Generate data quality dashboard

@performance-engineer: Please optimize data processing pipeline for speed and quality
```

### Model Training & Development

#### LoRA Training Implementation

```bash
@ai-engineer: Please implement LoRA training for Sesame CSM-1B:
1. Set up model architecture with LoRA adapters
2. Configure training hyperparameters for RTX 4090
3. Implement mixed precision training (bf16)
4. Add gradient checkpointing for memory efficiency
5. Create evaluation and validation loops

@mlops-engineer: Please set up training infrastructure:
1. Experiment tracking with MLflow or Weights & Biases
2. Automated checkpointing and recovery
3. Distributed training support (if needed)
4. Resource monitoring and alerting
5. Model artifact management
```

#### Model Evaluation

```bash
@data-scientist: Please implement comprehensive model evaluation:
1. Automated WER/CER calculation using Whisper
2. Speaker similarity measurement with ECAPA embeddings
3. Subjective quality assessment framework (MOS)
4. Inference speed benchmarking
5. Memory usage profiling

@performance-engineer: Please optimize model for production:
1. Quantization and pruning experiments
2. ONNX conversion for faster inference
3. Batch processing optimization
4. Memory usage optimization
5. Real-time performance validation
```

### API & Service Development

#### FastAPI Service

```bash
@api-designer: Please create production-ready API service:
1. RESTful endpoints for speech synthesis
2. Async request handling with queuing
3. Authentication and rate limiting
4. Comprehensive input validation
5. OpenAPI documentation and client SDKs

@backend-developer: Please implement service infrastructure:
1. Database integration for user management
2. File storage for audio artifacts
3. Caching layer for common requests
4. Monitoring and logging integration
5. Error handling and recovery
```

#### Episode Generation

```bash
@nlp-engineer: Please implement episode synthesis:
1. Script parsing (Markdown/JSON with SSML tags)
2. Text preprocessing and tokenization
3. Anchor insertion for identity stability
4. Timing and pacing control
5. Multi-paragraph continuity optimization

@ai-engineer: Please create audio post-processing pipeline:
1. Loudness normalization (-19 LUFS mono)
2. De-essing and noise reduction
3. Seamless concatenation of segments
4. Export to broadcast-ready formats
5. Metadata generation with timestamps
```

### Arbitrary Tool Usage & Artifacts

#### Using MCP Tools for Custom Workflows

```bash
# Generate specialized documentation
@technical-writer: Please use MCP tools to:
1. Generate API docs from code annotations
2. Create deployment guides with screenshots
3. Build interactive tutorials
4. Generate troubleshooting flowcharts
5. Create video documentation scripts

# Research and analysis
@research-analyst: Please use hugging-face MCP to:
1. Research latest TTS model architectures
2. Compare performance benchmarks
3. Identify potential model improvements
4. Generate competitive analysis report
5. Find relevant datasets and papers
```

#### Custom Artifact Generation

```bash
# Configuration and setup files
@platform-engineer: Please generate:
1. Kubernetes manifests for production deployment
2. Terraform modules for cloud infrastructure
3. Ansible playbooks for server configuration
4. GitHub Actions workflows for CI/CD
5. Monitoring and alerting configurations

# Development tools
@tooling-engineer: Please create:
1. Custom development scripts and utilities
2. Data validation and transformation tools
3. Performance profiling and benchmarking tools
4. Debugging and diagnostic utilities
5. Automated testing and QA tools
```

#### Integration with External Systems

```bash
# Home-K8s Integration
@platform-engineer: Please integrate with home-k8s:
1. Deploy MLflow for experiment tracking
2. Set up Prometheus monitoring integration
3. Configure ArgoCD for application deployment
4. Implement distributed storage for models
5. Create production inference services

# Third-party integrations
@api-designer: Please create integrations with:
1. Webhook endpoints for external notifications
2. Cloud storage services for data backup
3. Monitoring services for alerts
4. Authentication providers for security
5. Content management systems for scripts
```

---

## 🧠 Advanced Orchestration Patterns

### Complex Multi-Phase Projects

#### Orchestrating M1 Implementation

```bash
@workflow-orchestrator: Please design and execute M1 implementation workflow:

Phase 1: Data Foundation (Week 1)
- @data-engineer: Complete audio curation and processing
- @qa-expert: Validate data quality and coverage
- @performance-engineer: Optimize processing pipeline

Phase 2: Training Infrastructure (Week 2-3)
- @ai-engineer: Implement LoRA training pipeline
- @mlops-engineer: Set up monitoring and experiment tracking
- @devops-engineer: Create development environment

Phase 3: Model Development (Week 3-4)
- @ml-engineer: Execute training runs and hyperparameter tuning
- @data-scientist: Implement evaluation metrics and validation
- @performance-engineer: Optimize for RTX 4090 constraints

Phase 4: Integration & Testing (Week 4-5)
- @backend-developer: Create inference API service
- @test-automator: Implement comprehensive testing
- @api-designer: Design and document public APIs

Phase 5: Validation & Documentation (Week 5-6)
- @code-reviewer: Final code review and quality assurance
- @technical-writer: Create user documentation
- @performance-monitor: Validate all acceptance criteria
```

#### Cross-Team Collaboration

```bash
@multi-agent-coordinator: Please coordinate cross-functional team for podcast synthesis:

AI/ML Team:
- @ai-engineer: Core synthesis algorithms
- @ml-engineer: Training and optimization
- @nlp-engineer: Text processing and SSML

Infrastructure Team:
- @devops-engineer: Deployment and scaling
- @platform-engineer: Service integration
- @sre-engineer: Reliability and monitoring

Quality Team:
- @test-automator: Automated testing
- @performance-engineer: Performance optimization
- @qa-expert: Manual testing and validation

Documentation Team:
- @technical-writer: User guides and API docs
- @documentation-engineer: Technical specifications
- @api-documenter: Interactive documentation
```

### Error Handling & Recovery

#### Failure Recovery Workflows

```bash
@error-coordinator: Please handle training failure recovery:
1. Analyze failure logs and error patterns
2. Coordinate with @debugger to identify root cause
3. Work with @ai-engineer to implement fixes
4. Resume training from last valid checkpoint
5. Update monitoring to prevent similar failures

@workflow-orchestrator: Please implement resilient workflows:
1. Automatic retry mechanisms for transient failures
2. Checkpoint and recovery for long-running processes
3. Graceful degradation when services unavailable
4. Error notification and escalation procedures
5. Post-incident analysis and improvement
```

#### Performance Issues

```bash
@performance-monitor: Please coordinate performance investigation:
1. Identify performance bottlenecks in training pipeline
2. Work with @performance-engineer to profile critical paths
3. Coordinate with @ai-engineer to implement optimizations
4. Validate improvements with benchmarking
5. Update performance baselines and monitoring

@sre-engineer: Please ensure system reliability:
1. Implement circuit breakers for external dependencies
2. Set up automated scaling for resource-intensive tasks
3. Create runbooks for common operational issues
4. Establish SLO/SLI monitoring and alerting
5. Plan capacity for production workloads
```

### Custom Agent Combinations

#### Specialized Team Assembly

```bash
@agent-organizer: Please assemble specialized team for voice quality optimization:
- @audio-engineer: Audio processing expertise
- @ml-engineer: Model optimization knowledge
- @performance-engineer: Real-time processing skills
- @data-scientist: Quality metrics and evaluation
- @test-automator: Automated quality testing

@task-distributor: Please coordinate parallel development streams:
Stream A: @python-pro + @ai-engineer (Core algorithms)
Stream B: @backend-developer + @api-designer (Service layer)
Stream C: @devops-engineer + @platform-engineer (Infrastructure)
Stream D: @test-automator + @qa-expert (Quality assurance)
```

#### Dynamic Team Reconfiguration

```bash
@multi-agent-coordinator: Please adapt team composition based on current needs:
- If training issues: Add @debugger and @performance-engineer
- If quality problems: Add @qa-expert and @code-reviewer
- If integration challenges: Add @platform-engineer and @sre-engineer
- If performance bottlenecks: Add @performance-engineer and @optimization-specialist
- If documentation gaps: Add @technical-writer and @documentation-engineer
```

---

## 🔍 Troubleshooting & Recovery

### Common Issues & Solutions

#### System Startup Problems

```bash
# MCP Server connectivity issues
@error-coordinator: Please diagnose MCP server connectivity:
1. Check Redis server status and port availability
2. Verify Temporal server is running and accessible
3. Test MCP server connections individually
4. Review configuration files for correct endpoints
5. Restart services in correct dependency order

# Environment issues
@platform-engineer: Please resolve environment problems:
1. Verify Python 3.13+ and uv installation
2. Check CUDA drivers and PyTorch GPU access
3. Validate package dependencies and versions
4. Ensure proper virtual environment activation
5. Test basic functionality with hello-world examples
```

#### Training & Model Issues

```bash
# Memory exhaustion
@performance-engineer: Please resolve VRAM issues:
1. Implement gradient checkpointing
2. Reduce batch size and increase accumulation steps
3. Use mixed precision training (bf16)
4. Optimize model architecture for memory efficiency
5. Monitor and profile memory usage patterns

# Training convergence problems
@ai-engineer: Please debug training convergence:
1. Analyze loss curves and training metrics
2. Check learning rate scheduling and warmup
3. Verify data quality and preprocessing
4. Adjust hyperparameters based on validation metrics
5. Implement early stopping and regularization

# Data pipeline issues
@data-engineer: Please resolve data problems:
1. Validate audio format consistency (24kHz, 16-bit, mono)
2. Check dataset splits and metadata integrity
3. Verify text transcript accuracy and alignment
4. Test data loading performance and batching
5. Implement data validation and quality checks
```

#### Service & API Issues

```bash
# API performance problems
@backend-developer: Please optimize API performance:
1. Implement async request handling
2. Add caching for common requests
3. Optimize database queries and connections
4. Profile and optimize inference pipeline
5. Implement proper error handling and timeouts

# Integration failures
@platform-engineer: Please resolve integration issues:
1. Test service connectivity and authentication
2. Verify API contracts and data formats
3. Check timeout and retry configurations
4. Validate error handling and recovery procedures
5. Monitor service dependencies and health checks
```

### Debugging Workflows

#### Agent Coordination Issues

```bash
@error-coordinator: Please debug agent coordination problems:
1. Check task queue status and message delivery
2. Verify agent configurations and capabilities
3. Test communication between meta-orchestration agents
4. Validate workflow state and execution history
5. Review error logs for coordination failures

@context-manager: Please restore system state consistency:
1. Synchronize context between agents
2. Restore workflow state from checkpoints
3. Reconcile data and metadata integrity
4. Update agent knowledge bases
5. Verify system health after recovery
```

#### Performance Debugging

```bash
@performance-monitor: Please diagnose performance issues:
1. Profile CPU, GPU, and memory usage patterns
2. Identify bottlenecks in critical code paths
3. Analyze I/O and network performance
4. Monitor service response times and throughput
5. Generate performance reports and recommendations

@debugger: Please investigate complex bugs:
1. Reproduce issues in controlled environment
2. Analyze stack traces and error patterns
3. Use debugging tools and profilers
4. Implement targeted fixes and tests
5. Validate fixes don't introduce regressions
```

### Recovery Procedures

#### System Recovery

```bash
# Complete system restart
@platform-engineer: Please perform clean system restart:
1. Stop all MCP servers and services gracefully
2. Clear temporary files and reset state
3. Restart Redis and Temporal servers
4. Verify all MCP connections are healthy
5. Run system health checks and basic tests

# Data recovery
@data-engineer: Please recover from data corruption:
1. Restore from known good backup or checkpoint
2. Validate data integrity and format
3. Regenerate derived data and metadata
4. Update data processing pipeline if needed
5. Implement additional data validation checks
```

#### Workflow Recovery

```bash
# Resume interrupted workflows
@workflow-orchestrator: Please resume interrupted workflows:
1. Identify last successful checkpoint or state
2. Restore workflow context and variables
3. Resume execution from appropriate point
4. Monitor for successful completion
5. Update workflow design to prevent similar failures

# Agent state recovery
@context-manager: Please restore agent coordination:
1. Reset agent states to last known good configuration
2. Rebuild context and knowledge bases
3. Re-establish communication channels
4. Validate agent capabilities and assignments
5. Resume coordinated work from stable state
```

---

## 📚 Command Reference & Templates

### Quick Start Commands

#### System Activation

```bash
# Essential startup sequence
redis-server --daemonize yes
temporal server start-dev --ui-port 8080 --db-filename temporal.db
uv sync --all-extras
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Health check
redis-cli ping && echo "Redis OK"
curl -s http://localhost:8080 > /dev/null && echo "Temporal OK"
```

#### Agent Activation Templates

```bash
# Single agent
@[agent-name]: [Task description with specific requirements]

# Multi-agent coordination
@multi-agent-coordinator: Please coordinate [task] using [specific agents] to achieve [outcome]

# Workflow execution
@workflow-orchestrator: Please design and execute workflow for [process] with phases: [list phases]

# Task distribution
@task-distributor: Please allocate work for [goal] to appropriate specialists with priority [P0/P1/P2]
```

### Common Request Patterns

#### Development Tasks

```bash
# Code implementation
@[agent]: Please implement [feature/component] with requirements:
- [Requirement 1]
- [Requirement 2]
- [Requirement 3]
Following [specific standards/patterns]

# Code review
@code-reviewer: Please review [component/PR] for:
- Code quality and standards
- Performance optimization
- Error handling
- Security best practices
- Documentation completeness

# Testing
@test-automator: Please create tests for [component] including:
- Unit tests for core functions
- Integration tests for workflows
- Performance benchmarks
- Edge case validation
- Coverage reporting
```

#### Infrastructure & DevOps

```bash
# Environment setup
@devops-engineer: Please create [development/production] environment with:
- [Technology stack requirements]
- [Performance requirements]
- [Security requirements]
- [Monitoring requirements]
- [Documentation requirements]

# Deployment
@platform-engineer: Please deploy [application/service] to [environment] with:
- [Resource specifications]
- [Scaling requirements]
- [Monitoring setup]
- [Backup/recovery procedures]
- [Security configurations]
```

#### Quality & Validation

```bash
# Quality assurance
@qa-expert: Please validate [component/system] against:
- [Functional requirements]
- [Performance requirements]
- [Security requirements]
- [Usability requirements]
- [Reliability requirements]

# Performance optimization
@performance-engineer: Please optimize [component] for:
- [Performance targets]
- [Resource constraints]
- [Scalability requirements]
- [Memory efficiency]
- [Latency requirements]
```

### Workflow Templates

#### Feature Development Workflow

```bash
@workflow-orchestrator: Please execute feature development workflow:

1. Planning Phase:
   - @architect-reviewer: Design system architecture
   - @task-distributor: Break down into development tasks
   - @performance-engineer: Define performance requirements

2. Implementation Phase:
   - @[specialist-agent]: Implement core functionality
   - @code-reviewer: Ongoing code review
   - @test-automator: Create test suite

3. Integration Phase:
   - @platform-engineer: Integration testing
   - @performance-engineer: Performance validation
   - @qa-expert: Quality assurance testing

4. Deployment Phase:
   - @devops-engineer: Production deployment
   - @monitoring-specialist: Setup monitoring
   - @documentation-engineer: Update documentation
```

#### Bug Resolution Workflow

```bash
@error-coordinator: Please coordinate bug resolution:

1. Investigation:
   - @debugger: Reproduce and analyze issue
   - @performance-monitor: Check system metrics
   - @context-manager: Review related changes

2. Resolution:
   - @[domain-expert]: Implement fix
   - @code-reviewer: Review fix quality
   - @test-automator: Create regression tests

3. Validation:
   - @qa-expert: Validate fix effectiveness
   - @performance-engineer: Check performance impact
   - @deployment-engineer: Deploy to production

4. Follow-up:
   - @knowledge-synthesizer: Document lessons learned
   - @process-improvement: Update procedures
   - @monitoring-specialist: Add preventive monitoring
```

### Emergency Response Templates

#### Critical Issue Response

```bash
@error-coordinator: URGENT - [Critical issue description]
Please coordinate immediate response:

1. Immediate Actions:
   - @sre-engineer: Assess system impact and implement temporary mitigation
   - @debugger: Begin root cause analysis
   - @communication-specialist: Notify stakeholders

2. Investigation:
   - @[domain-expert]: Deep dive investigation
   - @performance-monitor: Analyze system metrics
   - @security-engineer: Check for security implications (if applicable)

3. Resolution:
   - @[implementation-agent]: Implement fix
   - @test-automator: Validate fix under stress
   - @deployment-engineer: Coordinate emergency deployment

4. Recovery:
   - @sre-engineer: Monitor system recovery
   - @data-engineer: Validate data integrity
   - @performance-engineer: Confirm performance restoration
```

#### System Recovery Template

```bash
@multi-agent-coordinator: Please coordinate system recovery:

1. Assessment:
   - @platform-engineer: Assess system state and damage
   - @data-engineer: Check data integrity
   - @security-engineer: Verify no security compromise

2. Recovery:
   - @sre-engineer: Execute recovery procedures
   - @backup-specialist: Restore from backups if needed
   - @network-engineer: Verify connectivity and access

3. Validation:
   - @test-automator: Run comprehensive system tests
   - @performance-engineer: Validate performance baselines
   - @qa-expert: Confirm all functionality restored

4. Post-Recovery:
   - @incident-analyst: Conduct post-incident review
   - @process-improvement: Update recovery procedures
   - @documentation-engineer: Update runbooks and documentation
```

---

## 🎓 Best Practices & Tips

### Effective Agent Communication

#### Clear Task Specification

```bash
# Good: Specific and actionable
@ai-engineer: Please implement LoRA training for Sesame CSM-1B with rank=16, batch_size=2, mixed precision bf16, targeting ≤22GB VRAM on RTX 4090

# Avoid: Vague and unclear
@ai-engineer: Make the training work better
```

#### Proper Context Sharing

```bash
# Good: Provides context and constraints
@performance-engineer: Please optimize the training pipeline. Current setup: RTX 4090 with 24GB VRAM, targeting 100 steps/hour, experiencing memory overflow at batch_size=4. Project constraints: must complete training in ≤4 hours for 30-minute dataset.

# Include relevant files and data
@code-reviewer: Please review training.py (lines 45-120) for memory efficiency. Current VRAM usage exceeds 22GB limit. Related files: config.yaml, data_loader.py
```

#### Coordination Patterns

```bash
# Use coordinators for complex tasks
@multi-agent-coordinator: Please coordinate implementation of evaluation framework requiring expertise from @data-scientist (metrics), @ml-engineer (pipeline), and @performance-engineer (optimization)

# Direct assignment for simple tasks
@test-automator: Please create unit tests for the audio_processor.py module
```

### Resource Management

#### GPU Memory Management

```bash
@performance-engineer: Please implement GPU memory optimization:
- Monitor VRAM usage with nvidia-smi integration
- Implement gradient checkpointing
- Use mixed precision training (bf16)
- Clear cache between training runs
- Profile memory allocation patterns
```

#### Parallel Work Coordination

```bash
@task-distributor: Please coordinate parallel development avoiding conflicts:
- Stream A: @data-engineer working on dataset optimization (data/ directory)
- Stream B: @ai-engineer working on model architecture (models/ directory)
- Stream C: @devops-engineer working on infrastructure (docker/, k8s/)
- Stream D: @test-automator working on tests (tests/ directory)
```

### Quality Assurance

#### Progressive Quality Gates

```bash
# Phase 1: Basic validation
@code-reviewer: Please check basic code quality (syntax, formatting, type hints)

# Phase 2: Functional validation
@test-automator: Please validate functionality with unit and integration tests

# Phase 3: Performance validation
@performance-engineer: Please validate performance meets requirements

# Phase 4: Security and compliance
@security-engineer: Please validate security best practices and compliance
```

#### Continuous Improvement

```bash
@knowledge-synthesizer: Please extract lessons learned from [project/sprint] and update:
- Best practices documentation
- Common pitfalls and solutions
- Successful patterns and templates
- Tool usage recommendations
- Performance optimization techniques
```

---

## 🔄 Getting Back to Work

### Resuming from Current State

We're currently at the perfect stopping point with infrastructure complete and ready for M1 implementation. Here's how to resume:

#### Immediate Next Steps

```bash
# 1. Verify system is ready
@multi-agent-coordinator: Please verify our M1 readiness by checking:
- MCP servers are operational (temporal, redis, taskqueue, etc.)
- Development environment is properly configured
- voice-dataset-kit is functional and tested
- Agent coordination system is responsive

# 2. Begin audio curation
@data-engineer: Please help me curate 30-90 minutes of target speaker audio:
- Analyze provided source files for quality and consistency
- Run voice-dataset-kit processing with optimal settings
- Perform quality review and validation
- Generate training/validation splits with proper metadata

# 3. Coordinate M1 implementation
@workflow-orchestrator: Please design and begin executing M1 implementation workflow following the plan in project_documentation/M1_IMPLEMENTATION_PLAN.md
```

#### Picking Up Specific Tasks

```bash
# Resume development work
@task-distributor: Please check taskqueue for current project status and assign next priority tasks

# Continue where we left off
@context-manager: Please restore project context and coordinate continuation of work from current M1 ready state

# Address any issues
@error-coordinator: Please check for any system issues or blockers that need resolution before proceeding with M1
```

This comprehensive guide provides everything needed to effectively orchestrate the 30+ agent system for any task, from simple code improvements to complex multi-phase project implementation. The key is clear communication, proper coordination, and systematic use of the available MCP servers and agent capabilities.

---

*This guide will be updated as we learn more about effective orchestration patterns and discover new use cases. Please share feedback and successful patterns with @knowledge-synthesizer for inclusion in future versions.*

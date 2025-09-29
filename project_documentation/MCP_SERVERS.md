# MCP Servers & Tools Documentation

## 📋 Overview

This document catalogs all MCP servers and tools required by the Claude Code subagents identified for the podcast pipeline project. The approach prioritizes local development with optional deployment to the home-k8s infrastructure when persistence or advanced monitoring is needed.

## 🎯 Local-First Development Strategy

- **Primary Environment**: Local development on developer machine
- **Infrastructure**: Optional deployment to `~/git/home-k8s` repo for persistence/monitoring
- **Tooling**: Modern Python ecosystem with uv, ruff, and hatchling
- **Deployment**: ArgoCD manages applications in home k8s cluster

## 🤖 Agent Tool Requirements

### Status Legend

- ✅ **Available**: Tool/server is implemented and ready to use
- 🔧 **Needs Setup**: Tool/server needs installation or configuration
- 🚫 **Not Needed**: Tool/server not required for this project
- ❌ **Not Available**: Tool/server not implemented and not planned

### Meta-Orchestration Agents (Priority: P1)

#### multi-agent-coordinator

- **Tools Required**: `Read`, `Write`, `message-queue`, `pubsub`, `workflow-engine`
- **Status**:
  - ✅ Read, Write: Available (built-in)
  - ✅ message-queue: RedisMCPServer implemented
  - ✅ pubsub: RedisMCPServer implemented
  - ✅ workflow-engine: temporal-mcp implemented
- **Setup**: Start Redis server (`redis-server`) and Temporal server (`temporal server start-dev)

#### workflow-orchestrator

- **Tools Required**: `Read`, `Write`, `workflow-engine`, `state-machine`, `bpmn`
- **Status**:
  - ✅ Read, Write: Available (built-in)
  - ✅ workflow-engine: temporal-mcp implemented
  - ✅ state-machine: Use Temporal workflows instead
  - 🚫 bpmn: Not needed for this project
- **Setup**: Start Temporal server (`temporal server start-dev`)

#### context-manager

- **Tools Required**: `Read`, `Write`, `redis`, `elasticsearch`, `vector-db`
- **Status**:
  - ✅ Read, Write: Available (built-in)
  - ✅ redis: RedisMCPServer implemented
  - 🚫 elasticsearch: Not needed for this project
  - 🚫 vector-db: Not needed for this project
- **Setup**: Start Redis server (`redis-server`)

#### task-distributor

- **Tools Required**: `Read`, `Write`, `task-queue`, `load-balancer`, `scheduler`
- **Status**:
  - ✅ Read, Write: Available (built-in)
  - ✅ task-queue: taskqueue MCP server implemented
  - 🚫 load-balancer: Not needed for this project
  - 🔧 scheduler: APScheduler or similar Python library
- **Setup**: Use taskqueue MCP server for task management

#### agent-organizer

- **Tools Required**: `Read`, `Write`, `agent-registry`, `task-queue`, `monitoring`
- **Status**:
  - ✅ Read, Write: Available (built-in)
  - 🔧 agent-registry: Simple JSON/YAML configuration
  - ✅ task-queue: taskqueue MCP server implemented
  - 🚫 monitoring: Not needed for this project (can use home-k8s if required)
- **Setup**: Use taskqueue MCP server and configuration files

#### performance-monitor

- **Tools Required**: `Read`, `Write`, `MultiEdit`, `Bash`, `prometheus`, `grafana`, `datadog`, `elasticsearch`, `statsd`
- **Status**:
  - ✅ Read, Write, MultiEdit, Bash: Available (built-in)
  - 🚫 prometheus, grafana: Not needed for this project (available in home-k8s if required)
  - 🚫 datadog: Not needed for this project
  - 🚫 elasticsearch: Not needed for this project
  - 🚫 statsd: Not needed for this project
- **Setup**: Use Python logging for local development

#### error-coordinator

- **Tools Required**: `Read`, `Write`, `MultiEdit`, `Bash`, `sentry`, `pagerduty`, `error-tracking`, `circuit-breaker`
- **Status**:
  - ✅ Read, Write, MultiEdit, Bash: Available (built-in)
  - 🔧 sentry: Can set up locally or use Sentry.io (optional)
  - 🚫 pagerduty: Not needed for this project
  - 🔧 error-tracking: Simple logging initially
  - 🔧 circuit-breaker: Python circuit breaker libraries
- **Setup**: Use Python logging and circuit breaker patterns

#### knowledge-synthesizer

- **Tools Required**: `Read`, `Write`, `MultiEdit`, `Bash`, `vector-db`, `nlp-tools`, `graph-db`, `ml-pipeline`
- **Status**:
  - ✅ Read, Write, MultiEdit, Bash: Available (built-in)
  - 🚫 vector-db: Not needed for this project
  - 🔧 nlp-tools: spaCy, NLTK, transformers
  - 🚫 graph-db: Not needed for this project
  - 🔧 ml-pipeline: Part of core project infrastructure
- **Setup**: Use transformers and NLP libraries for text processing

### Core Development Agents (Priority: P0)

#### python-pro

- **Tools Required**: `Read`, `Write`, `MultiEdit`, `Bash`, `uv`, `pytest`, `ruff`, `mypy`, `bandit`
- **Status**:
  - ✅ Read, Write, MultiEdit, Bash: Available (built-in)
  - ✅ uv: Need to install (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
  - ✅ pytest, ruff, mypy, bandit: Available via uv
- **Setup**: `uv add --dev pytest ruff mypy bandit`

#### ai-engineer

- **Tools Required**: `python`, `jupyter`, `tensorflow`, `pytorch`, `huggingface`, `wandb`
- **Status**:
  - ✅ python: Available (3.13)
  - 🔧 jupyter: Install via uv
  - 🔧 tensorflow: Install via uv if needed (project uses PyTorch)
  - ✅ pytorch: Available in pyproject.toml
  - 🔧 huggingface: Install transformers, datasets libraries
  - 🔧 wandb: For experiment tracking
- **Setup**: `uv add --dev jupyter transformers datasets wandb torch torchaudio`

#### ml-engineer

- **Tools Required**: `mlflow`, `kubeflow`, `tensorflow`, `sklearn`, `optuna`
- **Status**:
  - 🔧 mlflow: Can run locally or in home-k8s
  - ❌ kubeflow: Not needed for local development
  - 🔧 tensorflow: Install if needed
  - ✅ sklearn: Standard ML library
  - 🔧 optuna: Hyperparameter optimization
- **Setup**: `uv add --dev mlflow scikit-learn optuna`

#### data-engineer

- **Tools Required**: `spark`, `airflow`, `dbt`, `kafka`, `snowflake`, `databricks`
- **Status**:
  - ❌ spark: Overkill for local development
  - ❌ airflow: Too heavy for local development
  - 🔧 dbt: Can use for data transformations if needed
  - ❌ kafka: Not needed for local development
  - ❌ snowflake, databricks: Cloud services, not needed locally
- **Setup**: Simple Python scripts for data processing, no external dependencies needed

### Quality Assurance Agents (Priority: P1)

#### code-reviewer

- **Tools Required**: `Read`, `Grep`, `Glob`, `git`, `eslint`, `sonarqube`, `semgrep`
- **Status**:
  - ✅ Read, Grep, Glob, git: Available (built-in)
  - ❌ eslint: JavaScript tool, not needed for Python project
  - 🔧 sonarqube: Can run locally with Docker for advanced analysis
  - 🔧 semgrep: Static analysis tool for security and quality
- **Setup**: `uv add --dev semgrep` for static analysis

#### test-automator

- **Tools Required**: `Read`, `Write`, `selenium`, `cypress`, `playwright`, `pytest`, `jest`, `appium`, `k6`, `jenkins`
- **Status**:
  - ✅ Read, Write: Available (built-in)
  - ❌ selenium, cypress, playwright: Web testing tools, not needed for ML project
  - ✅ pytest: Available
  - ❌ jest: JavaScript testing, not needed
  - ❌ appium: Mobile testing, not needed
  - 🔧 k6: Load testing tool
  - ❌ jenkins: Using local development approach
- **Setup**: `pytest` with coverage for comprehensive testing

#### performance-engineer

- **Tools Required**: `Read`, `Grep`, `jmeter`, `gatling`, `locust`, `newrelic`, `datadog`, `prometheus`, `perf`, `flamegraph`
- **Status**:
  - ✅ Read, Grep: Available (built-in)
  - 🔧 jmeter, gatling, locust: Load testing tools (optional)
  - ❌ newrelic, datadog: Commercial APM tools
  - ✅ prometheus: Available in home-k8s
  - 🔧 perf, flamegraph: Profiling tools
- **Setup**: Python profiling tools (cProfile, py-spy) for local development

#### debugger

- **Tools Required**: `Read`, `Grep`, `Glob`, `gdb`, `lldb`, `chrome-devtools`, `vscode-debugger`, `strace`, `tcpdump`
- **Status**:
  - ✅ Read, Grep, Glob: Available (built-in)
  - 🔧 gdb, lldb: System debuggers (available on Linux)
  - ❌ chrome-devtools: Web debugging, not needed
  - ✅ vscode-debugger: IDE debugging capabilities
  - 🔧 strace, tcpdump: System debugging tools
- **Setup**: Use Python debugger (pdb, ipdb) and IDE debugging

### Infrastructure Agents (Priority: P2)

#### devops-engineer

- **Tools Required**: `Read`, `Write`, `MultiEdit`, `Bash`, `docker`, `kubernetes`, `terraform`, `ansible`, `prometheus`, `jenkins`
- **Status**:
  - ✅ Read, Write, MultiEdit, Bash: Available (built-in)
  - ✅ docker: Available (`/usr/bin/docker`)
  - ❌ kubernetes: Local development approach, available in home-k8s
  - ❌ terraform: Not needed for local development
  - ❌ ansible: Not needed for local development
  - ✅ prometheus: Available in home-k8s
  - ❌ jenkins: Using local development approach
- **Setup**: Docker for containerization, deploy to home-k8s when needed

#### platform-engineer

- **Tools Required**: `Read`, `Write`, `MultiEdit`, `Bash`, `kubectl`, `helm`, `argocd`, `crossplane`, `backstage`, `terraform`, `flux`
- **Status**:
  - ✅ Read, Write, MultiEdit, Bash: Available (built-in)
  - ✅ kubectl, helm, argocd: Available in home-k8s environment
  - 🔧 crossplane: Available in home-k8s for infrastructure composition
  - 🔧 backstage: Developer portal (optional)
  - ❌ terraform: Not using for local development
  - 🔧 flux: GitOps tool (argocd is primary choice)
- **Setup**: Connect to home-k8s cluster when needed for deployment

### Specialized Domain Agents (Priority: P2)

#### nlp-engineer

- **Tools Required**: `Read`, `Write`, `MultiEdit`, `Bash`, `transformers`, `spacy`, `nltk`, `huggingface`, `gensim`, `fasttext`
- **Status**:
  - ✅ Read, Write, MultiEdit, Bash: Available (built-in)
  - ✅ transformers: Available via huggingface
  - 🔧 spacy: Advanced NLP library
  - 🔧 nltk: Natural language toolkit
  - ✅ huggingface: Model hub access
  - 🔧 gensim: Topic modeling
  - 🔧 fasttext: Text classification
- **Setup**: `uv add --dev transformers spacy nltk gensim`

#### prompt-engineer

- **Tools Required**: `openai`, `anthropic`, `langchain`, `promptflow`, `jupyter`
- **Status**:
  - 🔧 openai: API client for OpenAI models
  - 🔧 anthropic: API client for Claude
  - 🔧 langchain: LLM application framework
  - 🔧 promptflow: Microsoft prompt engineering tool
  - ✅ jupyter: Available for experimentation
- **Setup**: `uv add openai anthropic langchain` Add jupyter to --dev if not already present

## 🚀 Setup Priority & Implementation Plan

### Phase 0: Essential Local Development (Immediate)

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install core development dependencies
uv add --dev pytest ruff mypy bandit pre-commit

# Install ML/AI dependencies for voice processing
uv add torch torchaudio transformers datasets librosa soundfile

# Install experiment tracking
uv add --dev wandb mlflow

# Install data processing
uv add --dev pandas numpy scipy matplotlib seaborn

# Install web framework for APIs/UIs
uv add fastapi uvicorn pydantic
```

### Background Services Setup

For agents requiring external services, start these background processes:

```bash
# Redis server (required for RedisMCPServer) try without starting it first, as there may be a redis/valkey instance already available
$ redis-cli -u redis://localhost:6379
localhost:6379> ping
PONG
localhost:6379>
# Start redis-server only if necessary

# Temporal server (required for temporal-mcp)
temporal server start-dev

# Optional: Use tmux/screen to manage background services
tmux new-session -d -s redis 'redis-server'
tmux new-session -d -s temporal 'temporal server start-dev'
```

### Phase 1: Quality & Testing Tools (Week 1)

```bash
# Static analysis and security
uv add --dev semgrep safety

# Advanced testing
uv add --dev pytest-cov pytest-xdist hypothesis

# Performance profiling
uv add --dev py-spy memory-profiler line-profiler

# Documentation
uv add --dev mkdocs mkdocs-material
```

### Phase 2: Optional Infrastructure (Week 2-3)

```bash
# Local development services with Docker
docker run -d --name redis -p 6379:6379 redis:alpine
docker run -d --name mlflow -p 5000:5000 -v mlflow_data:/mlflow python:3.13 mlflow server --host 0.0.0.0

# Advanced NLP tools
uv add --dev spacy nltk gensim
uv run python -m spacy download en_core_web_sm

# Vector database for knowledge management
uv add chromadb faiss-cpu
```

### Phase 3: Home-K8s Integration (As Needed)

When persistent storage, monitoring, or team collaboration is needed:

1. **MLflow**: Deploy to home-k8s for experiment persistence
2. **Prometheus/Grafana**: Use existing monitoring stack
3. **Redis**: Use for caching and pub/sub
4. **ArgoCD**: Deploy applications to cluster

## 🔧 MCP Server Requirements

### Custom MCP Servers (Development Status)

1. **Workflow Engine MCP Server** ✅
   - Purpose: State machine and workflow orchestration
   - Implementation: temporal-mcp (implemented)
   - Status: Available - requires `temporal server start-dev`

2. **Message Queue MCP Server** ✅
   - Purpose: Inter-agent communication
   - Implementation: RedisMCPServer (implemented)
   - Status: Available - requires `redis-server`

3. **Task Queue MCP Server** ✅
   - Purpose: Task distribution and coordination
   - Implementation: taskqueue (implemented)
   - Status: Available

4. **ML Tools MCP Server** 🔧
   - Purpose: MLflow, model registry, experiment tracking
   - Implementation: MLflow API wrapper
   - Status: Not yet needed - can implement when M1 training phase requires it

### Available MCP Servers

1. **Notion MCP Server** ✅
   - Status: Already connected
   - Purpose: Documentation and knowledge management

2. **Hugging Face MCP Server** ✅
   - Status: Available
   - Purpose: Model and dataset discovery

3. **Canva MCP Server** ✅
   - Status: Connected
   - Purpose: Design and visual content creation

4. **Jam MCP Server** ✅
   - Status: Connected
   - Purpose: Bug reporting and issue tracking

5. **taskqueue MCP Server** ✅
   - Status: Implemented
   - Purpose: Task queue management and coordination

6. **temporal-mcp MCP Server** ✅
   - Status: Implemented
   - Purpose: Workflow orchestration and state management
   - Note: Requires Temporal server (`temporal server start-dev`)

7. **RedisMCPServer** ✅
   - Status: Implemented
   - Purpose: Message queuing, pub/sub, and caching
   - Note: Requires Redis server (`redis-server`)

8. **playwright MCP Server** ✅
   - Status: Implemented
   - Purpose: Web automation and testing

9. **github MCP Server** ✅
   - Status: Implemented
   - Purpose: GitHub repository management and operations

## 📊 Tool Availability Matrix

| Tool Category | Tool | Status | Setup Required | Priority |
|---------------|------|--------|----------------|----------|
| **Python Ecosystem** | uv | 🔧 | Install script | P0 |
| | ruff | ✅ | Via uv | P0 |
| | pytest | ✅ | Via uv | P0 |
| | mypy | ✅ | Via uv | P0 |
| | bandit | ✅ | Via uv | P0 |
| **ML/AI Tools** | pytorch | ✅ | In pyproject.toml | P0 |
| | transformers | 🔧 | Via uv | P0 |
| | wandb | 🔧 | Via uv | P1 |
| | mlflow | 🔧 | Via uv or home-k8s | P1 |
| **MCP Servers** | temporal-mcp | ✅ | Temporal server required | P1 |
| | RedisMCPServer | ✅ | Redis server required | P1 |
| | taskqueue | ✅ | Ready to use | P1 |
| | playwright | ✅ | Ready to use | P2 |
| | github | ✅ | Ready to use | P2 |
| | huggingface | ✅ | Ready to use | P1 |
| **Infrastructure** | docker | ✅ | System installed | P1 |
| | redis-server | 🔧 | System install or Docker | P1 |
| | temporal | 🔧 | Install temporal CLI | P1 |
| | prometheus | 🚫 | Not needed (available in home-k8s) | P2 |
| | grafana | 🚫 | Not needed (available in home-k8s) | P2 |
| **Quality Tools** | semgrep | 🔧 | Via uv | P1 |
| | pre-commit | ✅ | In pyproject.toml | P1 |
| **Development** | jupyter | 🔧 | Via uv | P1 |
| | git | ✅ | System installed | P0 |

## 🏠 Home-K8s Integration Points

The `~/git/home-k8s` repository provides production infrastructure for:

### Available Services

- **ArgoCD**: Application deployment and GitOps
- **Prometheus/Grafana**: Metrics and monitoring
- **OTEL**: Distributed tracing
- **Redis**: Caching and pub/sub messaging
- **Storage**: Persistent volumes for data

### Integration Strategy

1. **Local Development**: Use lightweight alternatives (Docker, file storage)
2. **Persistence Needed**: Deploy MLflow, databases to home-k8s
3. **Team Collaboration**: Use home-k8s for shared services
4. **Production Workloads**: Scale up model training in cluster

### Deployment Process

```bash
# From podcast-pipeline project
cd ~/git/home-k8s

# Create application manifests for podcast-pipeline
kubectl apply -f applications/podcast-pipeline/

# Monitor deployment via ArgoCD
# Access Grafana for monitoring
# Use cluster resources for training large models
```

## 📝 Next Steps

1. **Immediate**: Install uv and core Python dependencies
2. **Week 1**: Set up quality tools and testing infrastructure
3. **Week 2**: Add ML/AI tools for model training (M1 phase)
4. **Week 3**: Integrate with home-k8s for persistence and monitoring
5. **Ongoing**: Develop custom MCP servers as coordination needs grow

## 🔍 Notes

- **Local-First**: Prioritize tools that work well in local development
- **Kubernetes Optional**: Use home-k8s for persistence, monitoring, and scaling
- **Tool Evolution**: Start simple, add complexity as project grows
- **Agent Flexibility**: Agents adapt to available tools, graceful degradation when tools unavailable
- **Cost Efficiency**: Avoid commercial services where open-source alternatives exist
- **Development Velocity**: Choose tools that enhance rather than hinder development speed

---

*This document should be updated as new MCP servers are developed and integrated into the project.*

# Mock MCP Mode for CI/CD Testing

## Overview

The integration test suite can run in two modes using the new **MCP Client Abstraction Layer**:

1. **Real Mode** (default): Uses actual MCP servers via Claude Code's runtime environment
2. **Mock Mode**: Uses in-memory mocks for CI/CD environments without Claude Code

## Architecture: MCP Client Abstraction Layer

### Phase 1: Complete - Abstraction Layer

The new architecture uses a clean abstraction layer at `/home/gerald/git/podcast-pipeline/podcast_pipeline/infrastructure/mcp_client.py`:

```python
# Abstract base class
class MCPClient(ABC):
    """Defines all MCP operations."""
    async def redis_set(self, key: str, value: Any) -> dict[str, Any]: ...
    async def taskqueue_create_project(self, ...) -> dict[str, Any]: ...
    # ... all MCP operations

# Real implementation (calls Claude Code MCP functions)
class RealMCPClient(MCPClient):
    """Calls actual MCP server functions injected by Claude Code."""
    async def redis_set(self, key: str, value: Any) -> dict[str, Any]:
        if "mcp__RedisMCPServer__set" in globals():
            return mcp__RedisMCPServer__set(key=key, value=value)
        raise RuntimeError("MCP function not available")

# Mock implementation (uses in-memory state)
class MockMCPClient(MCPClient):
    """Uses mock implementations from tests/fixtures/mocks.py."""
    def __init__(self):
        self.redis = MockRedisMCP()
        self.temporal = MockTemporalMCP()
        self.taskqueue = MockTaskQueueMCP()

    async def redis_set(self, key: str, value: Any) -> dict[str, Any]:
        return await self.redis.set(key=key, value=value)

# Factory function
def create_mcp_client(use_mock: bool = False) -> MCPClient:
    """Create appropriate client based on mode."""
    if use_mock:
        return MockMCPClient()
    return RealMCPClient()
```

### Phase 2: Complete - Test Infrastructure

Updated test infrastructure at `/home/gerald/git/podcast-pipeline/tests/integration_tests/conftest.py`:

```python
import os
import pytest
from podcast_pipeline.infrastructure.mcp_client import MCPClient, create_mcp_client

USE_MOCK_MCP = os.environ.get("USE_MOCK_MCP", "0") == "1"

@pytest.fixture(scope="function")
async def mcp_client() -> AsyncGenerator[MCPClient, None]:
    """Provide MCP client (real or mock based on environment)."""
    client = create_mcp_client(use_mock=USE_MOCK_MCP)
    yield client

@pytest.fixture(scope="function")
async def reset_mcp_state(mcp_client: MCPClient) -> None:
    """Reset MCP state between tests (for mock mode)."""
    if USE_MOCK_MCP and hasattr(mcp_client, 'redis'):
        mcp_client.redis.data = {}
        mcp_client.redis.hashes = {}
        mcp_client.redis.pub_sub_messages = {}
    # ... reset other state
```

**Key Benefits**:
- ✅ Clean separation of concerns
- ✅ Type-safe async operations
- ✅ Easy to test (inject mock via fixture)
- ✅ Works with both `pytest` and Claude Code
- ✅ Function-scoped for test isolation
- ✅ Automatic state reset between tests

### Phase 3: Pending - Update Tests

Tests need to be updated to use the `mcp_client` fixture instead of direct MCP function calls:

**Before** (direct MCP calls - Phase 0):
```python
async def test_redis_operations(self):
    # Direct call - only works in Claude Code environment
    result = mcp__RedisMCPServer__set(key="test", value="data")
    assert result["status"] == "success"
```

**After** (using mcp_client fixture - Phase 3):
```python
async def test_redis_operations(self, mcp_client):
    # Uses fixture - works in both modes
    result = await mcp_client.redis_set(key="test", value="data")
    assert result["status"] == "success"
```

## Running Tests

### Method 1: Via Claude Code (Real MCP Servers)

Ask Claude Code to run tests:
```
"Run the integration tests"
```

Claude Code will:
1. Execute tests in its runtime environment
2. Inject MCP functions into global namespace
3. RealMCPClient calls these functions
4. Tests run against **real** MCP infrastructure

**Requirements**:
- Claude Code environment
- Redis/Valkey server running on localhost:6379
- MCP servers configured and running

**Check infrastructure**:
```bash
# Check Redis
redis-cli ping  # Should return: PONG

# Check MCP servers
ps aux | grep -E "(redis-mcp-server|taskqueue-mcp|temporal-mcp)"
```

### Method 2: Direct pytest (Mock Mode - After Phase 3)

```bash
# Mock mode (CI/CD friendly)
USE_MOCK_MCP=1 uv run pytest tests/integration_tests/ -v

# Real mode (requires Claude Code MCP functions)
uv run pytest tests/integration_tests/ -v
```

**Current Status**: Tests still use direct MCP calls (Phase 0), so direct pytest won't work until Phase 3 is complete.

## Environment Variables

### USE_MOCK_MCP

Controls which MCP client implementation is used:

- `USE_MOCK_MCP=0` (default): Uses `RealMCPClient`
- `USE_MOCK_MCP=1`: Uses `MockMCPClient`

```bash
# Enable mock mode
export USE_MOCK_MCP=1
uv run pytest tests/integration_tests/ -v

# Disable mock mode (use real MCP servers)
export USE_MOCK_MCP=0
uv run pytest tests/integration_tests/ -v
```

## Test Fixtures

### mcp_client

Provides an `MCPClient` instance (real or mock based on `USE_MOCK_MCP`).

```python
async def test_example(mcp_client):
    """Test using injected MCP client."""
    # Set a Redis key
    result = await mcp_client.redis_set(key="test", value="data")
    assert result["status"] == "success"

    # Get the key back
    result = await mcp_client.redis_get(key="test")
    assert result["value"] == "data"

    # Create a project
    result = await mcp_client.taskqueue_create_project(
        initial_prompt="Test project",
        tasks=[{"title": "Task 1", "description": "Do something"}]
    )
    assert "project_id" in result
```

**Features**:
- Function-scoped (new instance per test)
- Automatically chooses Mock vs Real based on environment
- Type-safe (returns `MCPClient` interface)
- Async-compatible

### reset_mcp_state

Automatically resets mock state between tests (only affects `MockMCPClient`).

```python
async def test_isolated(mcp_client, reset_mcp_state):
    """Test with clean state."""
    # State is reset before this test runs
    result = await mcp_client.redis_get(key="test")
    assert "error" in result  # Key doesn't exist (state was reset)
```

**Features**:
- Function-scoped
- Only affects mock mode
- Clears Redis, Temporal, and TaskQueue mock state
- Ensures test isolation

## Mock Implementations

Mock implementations are in `/home/gerald/git/podcast-pipeline/tests/fixtures/mocks.py`:

### MockRedisMCP

In-memory Redis operations:

```python
class MockRedisMCP:
    def __init__(self):
        self.data = {}          # Key-value storage
        self.hashes = {}        # Hash storage
        self.pub_sub_messages = {}  # Pub/sub storage

    async def set(self, key: str, value: Any, ...) -> dict[str, Any]:
        self.data[key] = value
        return {"status": "success"}

    async def get(self, key: str) -> dict[str, Any]:
        if key in self.data:
            return {"value": self.data[key]}
        return {"error": f"Key '{key}' not found"}
```

**Operations**:
- Key: `set`, `get`, `delete`
- Hash: `hset`, `hget`, `hgetall`
- List: `lpush`, `llen`
- JSON: `json_set`, `json_get`
- Pub/Sub: `publish`, `subscribe`

### MockTemporalMCP

Mock workflow operations:

```python
class MockTemporalMCP:
    def __init__(self):
        self.workflows = {}

    async def get_workflow_history(
        self, workflow_id: str, run_id: str | None = None
    ) -> dict[str, Any]:
        if workflow_id in self.workflows:
            return {"status": "completed", "history": [...]}
        return {"error": f"Workflow '{workflow_id}' not found"}
```

**Operations**:
- `get_workflow_history`: Retrieve workflow history
- `register_workflow`: Helper to add test workflows

### MockTaskQueueMCP

Mock task queue operations:

```python
class MockTaskQueueMCP:
    def __init__(self):
        self.projects = {}
        self.next_project_id = 1
        self.next_task_id = 1

    async def create_project(
        self, initial_prompt: str, tasks: list[dict], ...
    ) -> dict[str, Any]:
        project_id = f"proj-{self.next_project_id}"
        self.next_project_id += 1
        self.projects[project_id] = {...}
        return {"project_id": project_id}
```

**Operations**:
- `create_project`: Create new project
- `list_projects`: List all projects
- `read_project`: Get project details
- `add_tasks_to_project`: Add tasks to project
- `update_task`: Update task status
- `get_next_task`: Get next pending task

## Test Markers

### @pytest.mark.requires_real_mcp

Mark tests that require real MCP servers (skipped in mock mode):

```python
@pytest.mark.requires_real_mcp
async def test_complex_workflow(mcp_client):
    """This test requires real Temporal server."""
    # Test complex workflow that mocks can't simulate
    pass
```

### @pytest.mark.mock_compatible

Mark tests that work in both modes (informational):

```python
@pytest.mark.mock_compatible
async def test_redis_operations(mcp_client):
    """This test works with both real and mock Redis."""
    result = await mcp_client.redis_set(key="test", value="data")
    assert result["status"] == "success"
```

## Mock Limitations

### Current Limitations

1. **Redis JSON**: Simplified JSONPath support (only root "$" path fully supported)
2. **Temporal**: No actual workflow execution, only history retrieval mocks
3. **Pub/Sub**: No actual message delivery to subscribers (messages stored but not delivered)
4. **State Persistence**: All state is in-memory and lost between test runs

### When to Use Each Mode

**Use Real Mode When**:
- Testing actual MCP server integrations
- Validating Redis pub/sub message delivery
- Testing Temporal workflow execution
- Performance testing
- Running tests via Claude Code tools

**Use Mock Mode When**:
- Running CI/CD pipelines
- Testing on machines without MCP servers
- Unit testing coordination logic
- Fast feedback loops
- Testing in containerized environments

## Migration Progress

### Phase 1: ✅ Complete - Abstraction Layer

Created MCP client abstraction at `podcast_pipeline/infrastructure/mcp_client.py`:
- `MCPClient` abstract base class
- `RealMCPClient` implementation
- `MockMCPClient` implementation
- `create_mcp_client()` factory function

### Phase 2: ✅ Complete - Test Infrastructure

Updated test infrastructure:
- `conftest.py` with new fixtures
- `mcp_client` fixture for injection
- `reset_mcp_state` fixture for cleanup
- Environment variable control
- Test markers for mode control

### Phase 3: 🔄 Pending - Update Tests

Need to update test files to use fixtures:
- Replace direct MCP calls with `mcp_client` fixture
- Update test signatures to accept `mcp_client` parameter
- Ensure all async operations are properly awaited
- Add test markers where appropriate

**Example migration**:
```python
# Before
async def test_redis_operations(self):
    result = mcp__RedisMCPServer__set(key="test", value="data")

# After
async def test_redis_operations(self, mcp_client):
    result = await mcp_client.redis_set(key="test", value="data")
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test-mock:
    name: Integration Tests (Mock Mode)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run integration tests (mock mode)
        env:
          USE_MOCK_MCP: 1
        run: |
          uv run pytest tests/integration_tests/ -v

  test-real:
    name: Integration Tests (Real MCP)
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Start MCP servers
        run: |
          # Start MCP server processes
          # (requires Claude Code or standalone MCP servers)

      - name: Run integration tests (real mode)
        run: |
          uv run pytest tests/integration_tests/ -v
```

## Troubleshooting

### Tests fail with "MCP function not available"

**In Real Mode**: This means MCP functions are not injected into the global namespace.

**Solution**: Run tests via Claude Code, or ensure MCP servers are running and accessible.

### Tests pass in mock mode but fail with real MCP

**Check**:
1. Real MCP servers are running
2. MCP server versions are compatible
3. Network connectivity to MCP servers
4. Redis server is running (`redis-cli ping`)

### State pollution between tests

**In Mock Mode**: Ensure `reset_mcp_state` fixture is being used.

```python
async def test_example(mcp_client, reset_mcp_state):
    # State is automatically reset before this test
    pass
```

### Type checking errors

**Issue**: `mcp_client` fixture returns `MCPClient` abstract type.

**Solution**: This is correct. The abstract interface ensures your tests work with both implementations.

## Infrastructure Validation

### Check MCP Server Status

```bash
# Check Redis
redis-cli ping  # Should return: PONG

# Check running MCP servers
ps aux | grep -E "(redis-mcp-server|taskqueue-mcp|temporal-mcp)"

# Check Temporal server
ps aux | grep "temporal server"
```

### Expected Output

```
✅ Redis: OPERATIONAL (PONG)
✅ RedisMCPServer: Running (PID xxxxx)
✅ TaskQueue MCP: Running (PID xxxxx)
✅ Temporal MCP: Running (PID xxxxx)
✅ Temporal Server: Running (PID xxxxx)
```

## Related Documentation

- [MCP Client Implementation](/home/gerald/git/podcast-pipeline/podcast_pipeline/infrastructure/mcp_client.py) - Abstraction layer code
- [Mock Implementations](/home/gerald/git/podcast-pipeline/tests/fixtures/mocks.py) - Mock server implementations
- [Agent Coordination Hooks](/.claude/hooks/agent-coordination-overview.md) - Agent coordination patterns
- [Infrastructure Assessment](../../INFRASTRUCTURE_ASSESSMENT.md) - MCP server infrastructure

## Summary

**Current Status**: ✅ Phase 2 Complete - Test infrastructure ready with MCP client abstraction

**Architecture**:
- ✅ Abstraction layer implemented (`MCPClient`, `RealMCPClient`, `MockMCPClient`)
- ✅ Test fixtures implemented (`mcp_client`, `reset_mcp_state`)
- ✅ Environment control implemented (`USE_MOCK_MCP`)
- 🔄 Test migration pending (Phase 3)

**For Now**:
- ✅ Fixtures ready for use in new tests
- ✅ Mock mode fully functional for new tests
- 🔄 Existing tests need migration to use fixtures
- 📋 Run tests via Claude Code until Phase 3 complete

**Next Step**: Update test files to use `mcp_client` fixture (Phase 3)

---

**Last Updated**: 2025-09-30
**Status**: Phase 2 Complete - Infrastructure Ready
**Maintained By**: test-automator, devops-engineer

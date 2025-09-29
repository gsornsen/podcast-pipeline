# Integration Tests with Real MCP Server Calls

## Overview

This directory contains integration tests that use **real MCP (Model Context Protocol) server function calls** to validate the podcast pipeline's infrastructure. The tests are designed to work within the Claude Code environment where MCP functions are available as global functions.

## Test Structure

### Main Test File
- `/tests/integration_tests/test_integration_pipeline.py` - Complete integration test suite

### MCP Servers Tested
- **RedisMCPServer**: Redis operations for state management and communication
- **TaskQueue**: Project and task management operations
- **Temporal MCP**: Workflow history and execution tracking

## Real MCP Function Calls Implemented

### Redis MCP Functions
- `mcp__RedisMCPServer__set(key, value, expiration=None)` - Set key-value pairs
- `mcp__RedisMCPServer__get(key)` - Get value by key
- `mcp__RedisMCPServer__delete(key)` - Delete key
- `mcp__RedisMCPServer__hset(name, key, value, expire_seconds=None)` - Hash operations
- `mcp__RedisMCPServer__hget(name, key)` - Get hash field
- `mcp__RedisMCPServer__lpush(name, value, expire=None)` - List operations
- `mcp__RedisMCPServer__llen(name)` - Get list length

### TaskQueue MCP Functions
- `mcp__taskqueue__list_projects(state=None)` - List all projects
- `mcp__taskqueue__create_project(initialPrompt, tasks, autoApprove=None)` - Create new project
- `mcp__taskqueue__add_tasks_to_project(projectId, tasks)` - Add tasks to existing project
- `mcp__taskqueue__read_project(projectId)` - Read project details

### Temporal MCP Functions
- `mcp__temporal_mcp__GetWorkflowHistory(workflowId, runId=None)` - Get workflow execution history

## Test Coverage

### System Health Tests
✅ **test_system_health()** - Validates Redis, Temporal, and MCP server connectivity

### MCP Connectivity Tests
✅ **test_mcp_connectivity()** - Tests all MCP server operations and validates responses

### Redis Communication Tests
✅ **test_redis_communication()** - Comprehensive Redis operations (keys, hashes, lists)

### Task Queue Integration Tests
✅ **test_task_queue_integration()** - Project creation, task management, and retrieval

### Temporal Workflow Tests
✅ **test_temporal_workflow()** - Workflow history retrieval and validation

### Multi-Agent Collaboration Tests
✅ **test_multi_agent_collaboration()** - Agent coordination using Redis state management

### Full Pipeline Test
✅ **test_full_pipeline()** - End-to-end integration test running all scenarios

## Environment Requirements

### Claude Code Environment
- Tests require Claude Code environment where MCP functions are available as globals
- When running outside Claude Code, tests will gracefully skip with appropriate messaging

### MCP Server Configuration
Tests expect MCP servers to be configured in `~/.claude.json`:
- **RedisMCPServer**: `uvx --from redis-mcp-server@latest redis-mcp-server --url redis://localhost:6379`
- **temporal-mcp**: `/usr/local/bin/temporal-mcp --config /home/gerald/.config/temporal/config/development.yaml`
- **taskqueue**: Configured and operational

### Redis Server
- Redis server running on localhost:6379
- Verified operational with `redis-cli ping`

## Running Tests

### In Claude Code Environment (Recommended)
```bash
# Run all integration tests
uv run pytest tests/integration_tests/test_integration_pipeline.py -v

# Run specific test
uv run pytest tests/integration_tests/test_integration_pipeline.py::TestRealIntegrationPipeline::test_system_health -v
```

### Outside Claude Code Environment
Tests will automatically skip with message: \"MCP functions not available - this test requires Claude Code environment\"

## Test Architecture

### Error Handling
- Graceful fallback when MCP functions are unavailable
- Proper error reporting for MCP call failures
- Session-based test isolation using unique test session IDs

### Session Management
- Each test run uses unique session ID for data isolation
- Automatic cleanup of test data during teardown
- No interference between concurrent test runs

### MCP Function Safety
- `_safe_mcp_call()` wrapper handles NameError and other exceptions
- Consistent error response format across all MCP operations
- Detailed logging of success/failure for each operation

## Expected Test Behavior

### In Claude Code Environment
- All tests should run and validate real MCP infrastructure
- Redis operations should succeed if Redis server is running
- TaskQueue operations should succeed if TaskQueue MCP is configured
- Temporal operations may have limited functionality depending on workflow service

### Outside Claude Code Environment
- All tests skip gracefully
- No failures due to missing MCP functions
- Clear messaging about environment requirements

## Integration with Podcast Pipeline

These tests validate the foundational infrastructure needed for:
- Multi-agent coordination using Redis state management
- Task distribution and management via TaskQueue
- Workflow execution tracking with Temporal
- Real-time communication between pipeline components

## Troubleshooting

### Common Issues
1. **\"MCP function not available\"** - Running outside Claude Code environment
2. **Redis connection errors** - Redis server not running on localhost:6379
3. **TaskQueue errors** - TaskQueue MCP server not configured or running
4. **Temporal errors** - Temporal workflow service not available (expected for test workflows)

### Verification Steps
1. Verify Redis: `redis-cli ping` should return `PONG`
2. Verify Temporal: `curl localhost:8233` should return valid JSON
3. Check MCP configuration in `~/.claude.json`
4. Run tests in Claude Code environment

## Future Enhancements

- Add performance benchmarking for MCP operations
- Implement stress testing for concurrent MCP calls
- Add integration with additional MCP servers (playwright, hugging-face)
- Expand test coverage for error scenarios and edge cases
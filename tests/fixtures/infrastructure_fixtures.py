"""Pytest fixtures for infrastructure testing."""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

from tests.fixtures.mocks import MockRedisMCP, MockTaskQueueMCP, MockTemporalMCP


@pytest.fixture
def mock_redis_mcp(monkeypatch: pytest.MonkeyPatch) -> Generator[MockRedisMCP]:
    """Mock RedisMCPServer responses for CI.

    This fixture patches Redis MCP tool calls to return mock responses,
    allowing tests to run without a live Redis server.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Yields:
        MockRedisMCP instance for test assertions

    Example:
        ```python
        def test_redis_operations(mock_redis_mcp):
            # Test code that uses Redis MCP tools
            # The mock will intercept all Redis calls
            result = await redis_wrapper.set_key("test", "value")
            assert "test" in mock_redis_mcp.data
        ```
    """
    mock = MockRedisMCP()

    # Patch MCP tool calls to use mock
    # TODO: Replace with actual MCP tool patch paths once wrappers are implemented
    with (
        patch("infrastructure.mcp_wrappers.RedisWrapper.set", new=mock.set),
        patch("infrastructure.mcp_wrappers.RedisWrapper.get", new=mock.get),
        patch("infrastructure.mcp_wrappers.RedisWrapper.hset", new=mock.hset),
        patch("infrastructure.mcp_wrappers.RedisWrapper.hgetall", new=mock.hgetall),
        patch("infrastructure.mcp_wrappers.RedisWrapper.publish", new=mock.publish),
        patch("infrastructure.mcp_wrappers.RedisWrapper.subscribe", new=mock.subscribe),
    ):
        yield mock


@pytest.fixture
def mock_temporal_mcp(monkeypatch: pytest.MonkeyPatch) -> Generator[MockTemporalMCP]:
    """Mock temporal-mcp responses for CI.

    This fixture patches Temporal MCP tool calls to return mock responses,
    allowing tests to run without a live Temporal server.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Yields:
        MockTemporalMCP instance for test assertions

    Example:
        ```python
        def test_workflow_execution(mock_temporal_mcp):
            # Register a mock workflow
            mock_temporal_mcp.register_workflow("test-workflow", status="completed")

            # Test code that queries workflow status
            result = await temporal_wrapper.get_workflow_history("test-workflow")
            assert result["status"] == "completed"
        ```
    """
    mock = MockTemporalMCP()

    # Patch MCP tool calls to use mock
    # TODO: Replace with actual MCP tool patch paths once wrappers are implemented
    with patch(
        "infrastructure.mcp_wrappers.TemporalWrapper.get_workflow_history",
        new=mock.get_workflow_history,
    ):
        yield mock


@pytest.fixture
def mock_taskqueue_mcp(monkeypatch: pytest.MonkeyPatch) -> Generator[MockTaskQueueMCP]:
    """Mock taskqueue MCP responses for CI.

    This fixture patches TaskQueue MCP tool calls to return mock responses,
    allowing tests to run without the taskqueue MCP server.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Yields:
        MockTaskQueueMCP instance for test assertions

    Example:
        ```python
        def test_task_distribution(mock_taskqueue_mcp):
            # Test code that creates projects and tasks
            result = await taskqueue_wrapper.create_project(
                "Test project",
                [{"title": "Task 1", "description": "Test task"}]
            )
            assert result["project_id"] in mock_taskqueue_mcp.projects
        ```
    """
    mock = MockTaskQueueMCP()

    # Patch MCP tool calls to use mock
    # TODO: Replace with actual MCP tool patch paths once wrappers are implemented
    with (
        patch(
            "infrastructure.mcp_wrappers.TaskQueueWrapper.create_project",
            new=mock.create_project,
        ),
        patch(
            "infrastructure.mcp_wrappers.TaskQueueWrapper.list_projects",
            new=mock.list_projects,
        ),
        patch(
            "infrastructure.mcp_wrappers.TaskQueueWrapper.add_tasks",
            new=mock.add_tasks_to_project,
        ),
        patch(
            "infrastructure.mcp_wrappers.TaskQueueWrapper.update_task",
            new=mock.update_task,
        ),
        patch(
            "infrastructure.mcp_wrappers.TaskQueueWrapper.get_next_task",
            new=mock.get_next_task,
        ),
    ):
        yield mock


@pytest.fixture
def mock_infrastructure(
    mock_redis_mcp: MockRedisMCP,
    mock_temporal_mcp: MockTemporalMCP,
    mock_taskqueue_mcp: MockTaskQueueMCP,
) -> dict[str, Any]:
    """Combined mock for all infrastructure components.

    Args:
        mock_redis_mcp: Redis mock fixture
        mock_temporal_mcp: Temporal mock fixture
        mock_taskqueue_mcp: TaskQueue mock fixture

    Returns:
        Dictionary of all mocks for easy access

    Example:
        ```python
        def test_full_pipeline(mock_infrastructure):
            redis = mock_infrastructure["redis"]
            temporal = mock_infrastructure["temporal"]
            taskqueue = mock_infrastructure["taskqueue"]

            # Test multi-component workflows
            # ...
        ```
    """
    return {
        "redis": mock_redis_mcp,
        "temporal": mock_temporal_mcp,
        "taskqueue": mock_taskqueue_mcp,
    }


@pytest.fixture
def mock_gpu_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock GPU availability check.

    This fixture makes GPU health checks always pass,
    allowing tests to run on machines without GPUs.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Example:
        ```python
        def test_training_pipeline(mock_gpu_available):
            # Test code that requires GPU
            # The mock will make GPU checks pass
            # ...
        ```
    """
    # TODO: Patch actual GPU check function once implemented
    async def mock_check_gpu(required_model: str = "RTX 4090") -> bool:
        return True

    monkeypatch.setattr(
        "infrastructure.health_checks.check_gpu_health",
        mock_check_gpu,
    )

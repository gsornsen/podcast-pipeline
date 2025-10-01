"""Optional helper fixtures for MCP testing.

This module provides additional helper fixtures beyond the core mcp_client fixture
in conftest.py. These are convenience fixtures for common test scenarios.
"""

import pytest
from typing import Any


@pytest.fixture
def sample_redis_data() -> dict[str, str]:
    """Provide sample Redis data for testing.

    Returns:
        Dictionary with sample key-value pairs
    """
    return {
        "test_key_1": "test_value_1",
        "test_key_2": "test_value_2",
        "config:app": "production",
        "status:health": "ok",
    }


@pytest.fixture
def sample_project_tasks() -> list[dict[str, str]]:
    """Provide sample task list for project creation.

    Returns:
        List of task dictionaries with title and description
    """
    return [
        {
            "title": "Setup test environment",
            "description": "Initialize test environment with required dependencies",
        },
        {
            "title": "Execute integration tests",
            "description": "Run full integration test suite",
        },
        {
            "title": "Generate test report",
            "description": "Create comprehensive test execution report",
        },
        {
            "title": "Cleanup test data",
            "description": "Remove temporary test data and restore environment",
        },
    ]


@pytest.fixture
def sample_workflow_data() -> dict[str, Any]:
    """Provide sample workflow data for Temporal testing.

    Returns:
        Dictionary with workflow configuration
    """
    return {
        "workflow_id": "test-workflow-001",
        "workflow_type": "VoiceCloneTraining",
        "task_queue": "training-tasks",
        "input": {
            "dataset_path": "/data/processed/test",
            "model_config": {"lora_rank": 16, "learning_rate": 1e-4},
        },
    }


@pytest.fixture
async def populated_redis_client(mcp_client, sample_redis_data):
    """Provide an MCP client with pre-populated Redis data.

    This fixture populates Redis with sample data before the test runs.
    Useful for testing read operations without setting up data in each test.

    Args:
        mcp_client: MCP client fixture
        sample_redis_data: Sample data fixture

    Yields:
        MCP client with populated Redis data
    """
    # Populate Redis with sample data
    for key, value in sample_redis_data.items():
        await mcp_client.redis_set(key=key, value=value)

    yield mcp_client

    # Cleanup is handled by reset_mcp_state fixture


@pytest.fixture
async def test_project(mcp_client, sample_project_tasks):
    """Create a test project and return its ID.

    This fixture creates a project in the task queue that can be used
    for testing project-related operations.

    Args:
        mcp_client: MCP client fixture
        sample_project_tasks: Sample tasks fixture

    Yields:
        Project ID string
    """
    result = await mcp_client.taskqueue_create_project(
        initial_prompt="Integration test project",
        tasks=sample_project_tasks,
        auto_approve=False,
    )

    project_id = result.get("project_id")
    yield project_id

    # Cleanup is handled by reset_mcp_state fixture


@pytest.fixture
def test_session_id() -> str:
    """Generate a unique test session ID.

    Returns:
        UUID string for test session identification
    """
    import uuid

    return str(uuid.uuid4())[:8]

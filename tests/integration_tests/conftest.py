"""Pytest configuration for integration tests with MCP client abstraction.

This module provides fixtures to enable mock or real MCP operations when running
integration tests, using the new MCPClient abstraction layer.

Set environment variable USE_MOCK_MCP=1 to enable mock mode.
"""

import os

import pytest

# Import the MCP client abstraction
from infrastructure.mcp_client import (
    MCPClient,
    create_mcp_client,
)

# Import helper fixtures (makes them available to all tests in this directory)

# Check if we should use mocks (CI/CD mode)
USE_MOCK_MCP = os.environ.get("USE_MOCK_MCP", "0") == "1"


@pytest.fixture(scope="function")
def mcp_client() -> MCPClient:
    """Provide MCP client (real or mock based on environment).

    This fixture creates an MCPClient instance that will use either:
    - MockMCPClient (when USE_MOCK_MCP=1) for CI/CD testing
    - RealMCPClient (default) for testing with actual MCP servers

    The fixture is function-scoped to ensure test isolation.

    Returns:
        MCPClient instance (either Mock or Real)

    Example:
        async def test_redis_operations(mcp_client):
            result = await mcp_client.redis_set(key="test", value="data")
            assert result["status"] == "success"
    """
    client = create_mcp_client(use_mock=USE_MOCK_MCP)

    if USE_MOCK_MCP:
        print("\n" + "="*80)
        print("🔧 MOCK MODE ENABLED - Using MockMCPClient for CI/CD testing")
        print("="*80)
        print("✅ Redis MCP: Mock implementation active")
        print("✅ Temporal MCP: Mock implementation active")
        print("✅ TaskQueue MCP: Mock implementation active")
        print("="*80 + "\n")

    return client


@pytest.fixture(scope="function", autouse=True)
def reset_mcp_state(mcp_client: MCPClient) -> None:
    """Reset MCP state between tests (for mock mode).

    This fixture ensures that mock MCP state is cleared between tests
    to prevent test pollution and ensure test isolation.

    Args:
        mcp_client: Fixture providing MCP client instance

    Note:
        This only affects MockMCPClient instances. RealMCPClient is not affected.
        This fixture runs automatically before each test (autouse=True).
    """
    if not USE_MOCK_MCP:
        return

    # Reset mock state if using MockMCPClient
    if hasattr(mcp_client, 'redis'):
        # Reset Redis mock state
        mcp_client.redis.data = {}
        mcp_client.redis.hashes = {}
        mcp_client.redis.pub_sub_messages = {}

    if hasattr(mcp_client, 'temporal'):
        # Reset Temporal mock state
        mcp_client.temporal.workflows = {}

    if hasattr(mcp_client, 'taskqueue'):
        # Reset TaskQueue mock state
        mcp_client.taskqueue.projects = {}
        mcp_client.taskqueue.next_project_id = 1
        mcp_client.taskqueue.next_task_id = 1


@pytest.fixture(scope="session", autouse=True)
def print_test_mode() -> None:
    """Print test mode information at session start."""
    mode = "MOCK MODE (CI/CD)" if USE_MOCK_MCP else "REAL MODE (MCP Servers)"
    print("\n" + "="*80)
    print(f"🧪 Integration Tests - {mode}")
    print("="*80)
    if USE_MOCK_MCP:
        print("📝 Using MockMCPClient - no real MCP servers required")
        print("💡 To use real MCP servers: unset USE_MOCK_MCP or set to 0")
    else:
        print("🔌 Using RealMCPClient - requires actual MCP servers")
        print("💡 To use mock mode: export USE_MOCK_MCP=1")
    print("="*80 + "\n")


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers.

    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers",
        "requires_real_mcp: mark test as requiring real MCP servers (skip in mock mode)"
    )
    config.addinivalue_line(
        "markers",
        "mock_compatible: mark test as compatible with mock MCP servers"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection based on mock mode.

    Args:
        config: Pytest configuration object
        items: List of collected test items
    """
    if not USE_MOCK_MCP:
        return

    # Skip tests that require real MCP servers when in mock mode
    skip_real_mcp = pytest.mark.skip(reason="Requires real MCP servers (USE_MOCK_MCP=1)")
    for item in items:
        if "requires_real_mcp" in item.keywords:
            item.add_marker(skip_real_mcp)

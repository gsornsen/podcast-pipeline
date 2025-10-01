"""Example test demonstrating the new MCP client fixture approach.

This test file shows how to write tests using the new MCPClient abstraction
that work in both mock and real modes.

Run in mock mode:
    USE_MOCK_MCP=1 uv run pytest tests/integration_tests/test_mcp_client_example.py -v

Run in real mode (requires Claude Code):
    uv run pytest tests/integration_tests/test_mcp_client_example.py -v
"""

import pytest


@pytest.mark.asyncio
@pytest.mark.mock_compatible
async def test_redis_basic_operations(mcp_client, reset_mcp_state):
    """Test basic Redis operations using MCP client fixture.

    This test demonstrates:
    - Using mcp_client fixture for Redis operations
    - Async/await pattern
    - Proper error handling
    - Mock compatibility
    """
    # Set a key
    result = await mcp_client.redis_set(key="test_key", value="test_value")
    assert "status" in result or "error" not in result
    print(f"✅ Redis SET: {result}")

    # Get the key back
    result = await mcp_client.redis_get(key="test_key")
    assert result.get("value") == "test_value"
    print(f"✅ Redis GET: {result}")

    # Delete the key
    result = await mcp_client.redis_delete(key="test_key")
    assert "status" in result or "error" not in result
    print(f"✅ Redis DELETE: {result}")

    # Verify deletion
    result = await mcp_client.redis_get(key="test_key")
    assert "error" in result
    print(f"✅ Redis GET (after delete): {result}")


@pytest.mark.asyncio
@pytest.mark.mock_compatible
async def test_redis_hash_operations(mcp_client, reset_mcp_state):
    """Test Redis hash operations."""
    hash_name = "test_hash"

    # Set hash field
    result = await mcp_client.redis_hset(
        name=hash_name, key="field1", value="value1"
    )
    assert "status" in result or "error" not in result
    print(f"✅ Redis HSET: {result}")

    # Get hash field
    result = await mcp_client.redis_hget(name=hash_name, key="field1")
    assert result.get("value") == "value1"
    print(f"✅ Redis HGET: {result}")

    # Set another field
    await mcp_client.redis_hset(name=hash_name, key="field2", value="value2")

    # Get all hash fields
    result = await mcp_client.redis_hgetall(name=hash_name)
    fields = result.get("fields", {})
    assert "field1" in fields
    assert "field2" in fields
    print(f"✅ Redis HGETALL: {result}")


@pytest.mark.asyncio
@pytest.mark.mock_compatible
async def test_redis_list_operations(mcp_client, reset_mcp_state):
    """Test Redis list operations."""
    list_name = "test_list"

    # Push items to list
    result = await mcp_client.redis_lpush(name=list_name, value="item1")
    assert "status" in result or "length" in result
    print(f"✅ Redis LPUSH: {result}")

    await mcp_client.redis_lpush(name=list_name, value="item2")
    await mcp_client.redis_lpush(name=list_name, value="item3")

    # Get list length
    result = await mcp_client.redis_llen(name=list_name)
    assert result.get("length") == 3
    print(f"✅ Redis LLEN: {result}")


@pytest.mark.asyncio
@pytest.mark.mock_compatible
async def test_taskqueue_project_lifecycle(mcp_client, reset_mcp_state):
    """Test TaskQueue project creation and management."""
    # Create project
    result = await mcp_client.taskqueue_create_project(
        initial_prompt="Test project for example",
        tasks=[
            {"title": "Task 1", "description": "First task"},
            {"title": "Task 2", "description": "Second task"},
        ],
        auto_approve=False,
    )
    assert "project_id" in result or "error" not in result
    project_id = result.get("project_id")
    print(f"✅ Project created: {project_id}")

    # Read project
    result = await mcp_client.taskqueue_read_project(project_id=project_id)
    assert "project" in result or "error" not in result
    print(f"✅ Project details: {result}")

    # Add more tasks
    result = await mcp_client.taskqueue_add_tasks_to_project(
        project_id=project_id,
        tasks=[
            {"title": "Task 3", "description": "Third task"},
        ],
    )
    assert "status" in result or "error" not in result
    print(f"✅ Tasks added: {result}")

    # List projects
    result = await mcp_client.taskqueue_list_projects(state="open")
    assert "projects" in result or "error" not in result
    print(f"✅ Projects listed: {len(result.get('projects', []))} projects")


@pytest.mark.asyncio
@pytest.mark.mock_compatible
async def test_temporal_workflow_history(mcp_client, reset_mcp_state):
    """Test Temporal workflow history retrieval."""
    workflow_id = "test-workflow-example"

    # Get workflow history (will not exist in mock mode)
    result = await mcp_client.temporal_get_workflow_history(
        workflow_id=workflow_id
    )

    # In mock mode, workflow won't exist - this is expected
    assert "error" in result or "workflow_id" in result
    print(f"✅ Workflow history result: {result}")


@pytest.mark.asyncio
@pytest.mark.mock_compatible
async def test_redis_pub_sub(mcp_client, reset_mcp_state):
    """Test Redis pub/sub operations."""
    channel = "test_channel"
    message = "test message"

    # Subscribe to channel
    result = await mcp_client.redis_subscribe(channel=channel)
    assert "status" in result or "error" not in result
    print(f"✅ Redis SUBSCRIBE: {result}")

    # Publish message
    result = await mcp_client.redis_publish(channel=channel, message=message)
    assert "subscribers" in result or "error" not in result
    print(f"✅ Redis PUBLISH: {result}")


@pytest.mark.asyncio
@pytest.mark.mock_compatible
async def test_using_helper_fixtures(
    mcp_client, reset_mcp_state, sample_redis_data, test_session_id
):
    """Test using helper fixtures from mcp_fixtures.py."""
    # Use sample data to populate Redis
    for key, value in sample_redis_data.items():
        await mcp_client.redis_set(key=key, value=value)

    # Verify one of the keys
    result = await mcp_client.redis_get(key="config:app")
    assert result.get("value") == "production"

    print(f"✅ Test session: {test_session_id}")
    print("✅ Sample data loaded and verified")


@pytest.mark.asyncio
async def test_state_isolation(mcp_client, reset_mcp_state):
    """Test that state is isolated between tests.

    This test should have clean state (no data from previous tests).
    """
    # Try to get a key that should not exist
    result = await mcp_client.redis_get(key="test_key")
    assert "error" in result  # Key should not exist

    # Set a key for next test to verify isolation
    await mcp_client.redis_set(key="isolation_test", value="data")
    print("✅ State isolation verified")


@pytest.mark.asyncio
async def test_state_was_reset(mcp_client, reset_mcp_state):
    """Test that state from previous test was reset.

    The key 'isolation_test' from previous test should not exist.
    """
    result = await mcp_client.redis_get(key="isolation_test")
    assert "error" in result  # Key should not exist (was reset)
    print("✅ State reset between tests verified")


# Example of a test that requires real MCP (would be skipped in mock mode)
@pytest.mark.asyncio
@pytest.mark.requires_real_mcp
async def test_requires_real_infrastructure(mcp_client, reset_mcp_state):
    """Test that requires real MCP infrastructure.

    This test is automatically skipped when USE_MOCK_MCP=1.
    """
    # Complex operations that require real infrastructure
    result = await mcp_client.redis_set(key="real_test", value="real_value")
    assert "status" in result
    print("✅ Real infrastructure test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

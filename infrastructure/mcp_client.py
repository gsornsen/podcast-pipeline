"""MCP Client abstraction layer for testing and production.

This module provides a unified interface for MCP operations that works in both
production (with real MCP servers) and testing (with mock implementations).

The abstraction enables:
- Seamless switching between real and mock MCP servers
- Type-safe async operations
- Consistent error handling
- Easy pytest integration
"""

from abc import ABC, abstractmethod
from typing import Any


class MCPClient(ABC):
    """Abstract base class for MCP operations.

    This interface defines all MCP operations used by the podcast pipeline.
    Implementations provide either real MCP server calls or mock responses.
    """

    # Redis operations
    @abstractmethod
    async def redis_set(
        self,
        key: str,
        value: Any,
        expiration: int | None = None,
    ) -> dict[str, Any]:
        """Set a Redis key with optional expiration.

        Args:
            key: Redis key
            value: Value to store
            expiration: Optional expiration in seconds

        Returns:
            Response dictionary with status
        """
        pass

    @abstractmethod
    async def redis_get(self, key: str) -> dict[str, Any]:
        """Get a Redis key value.

        Args:
            key: Redis key

        Returns:
            Response dictionary with value or error
        """
        pass

    @abstractmethod
    async def redis_delete(self, key: str) -> dict[str, Any]:
        """Delete a Redis key.

        Args:
            key: Redis key

        Returns:
            Response dictionary with status
        """
        pass

    @abstractmethod
    async def redis_hset(
        self,
        name: str,
        key: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Set a hash field with optional expiration.

        Args:
            name: Hash name
            key: Field name
            value: Field value
            expire_seconds: Optional expiration in seconds

        Returns:
            Response dictionary with status
        """
        pass

    @abstractmethod
    async def redis_hget(self, name: str, key: str) -> dict[str, Any]:
        """Get a hash field value.

        Args:
            name: Hash name
            key: Field name

        Returns:
            Response dictionary with value or error
        """
        pass

    @abstractmethod
    async def redis_hgetall(self, name: str) -> dict[str, Any]:
        """Get all fields and values from a hash.

        Args:
            name: Hash name

        Returns:
            Response dictionary with fields or error
        """
        pass

    @abstractmethod
    async def redis_lpush(
        self,
        name: str,
        value: Any,
        expire: int | None = None,
    ) -> dict[str, Any]:
        """Push a value to a Redis list.

        Args:
            name: List name
            value: Value to push
            expire: Optional expiration in seconds

        Returns:
            Response dictionary with status and length
        """
        pass

    @abstractmethod
    async def redis_llen(self, name: str) -> dict[str, Any]:
        """Get the length of a Redis list.

        Args:
            name: List name

        Returns:
            Response dictionary with length
        """
        pass

    @abstractmethod
    async def redis_json_set(
        self,
        name: str,
        path: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Set a JSON value at path.

        Args:
            name: Redis key
            path: JSON path (e.g., "$" for root)
            value: JSON value
            expire_seconds: Optional expiration in seconds

        Returns:
            Response dictionary with status
        """
        pass

    @abstractmethod
    async def redis_json_get(
        self,
        name: str,
        path: str = "$",
    ) -> dict[str, Any]:
        """Get a JSON value at path.

        Args:
            name: Redis key
            path: JSON path (default: "$" for root)

        Returns:
            Response dictionary with value or error
        """
        pass

    @abstractmethod
    async def redis_publish(self, channel: str, message: str) -> dict[str, Any]:
        """Publish a message to a Redis channel.

        Args:
            channel: Channel name
            message: Message content

        Returns:
            Response dictionary with subscriber count
        """
        pass

    @abstractmethod
    async def redis_subscribe(self, channel: str) -> dict[str, Any]:
        """Subscribe to a Redis channel.

        Args:
            channel: Channel name

        Returns:
            Response dictionary with status
        """
        pass

    # TaskQueue operations
    @abstractmethod
    async def taskqueue_create_project(
        self,
        initial_prompt: str,
        tasks: list[dict[str, str]],
        auto_approve: bool = False,
    ) -> dict[str, Any]:
        """Create a new project with tasks.

        Args:
            initial_prompt: Project prompt
            tasks: Task list
            auto_approve: Auto-approve flag

        Returns:
            Response dictionary with project_id
        """
        pass

    @abstractmethod
    async def taskqueue_list_projects(self, state: str = "open") -> dict[str, Any]:
        """List projects by state.

        Args:
            state: Project state filter

        Returns:
            Response dictionary with projects list
        """
        pass

    @abstractmethod
    async def taskqueue_read_project(self, project_id: str) -> dict[str, Any]:
        """Read project details.

        Args:
            project_id: Project ID

        Returns:
            Response dictionary with project details or error
        """
        pass

    @abstractmethod
    async def taskqueue_add_tasks_to_project(
        self,
        project_id: str,
        tasks: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Add tasks to an existing project.

        Args:
            project_id: Project ID
            tasks: Task list

        Returns:
            Response dictionary with status
        """
        pass

    @abstractmethod
    async def taskqueue_update_task(
        self,
        project_id: str,
        task_id: str,
        status: str,
        completed_details: str | None = None,
    ) -> dict[str, Any]:
        """Update task status.

        Args:
            project_id: Project ID
            task_id: Task ID
            status: New status
            completed_details: Completion details if done

        Returns:
            Response dictionary with status
        """
        pass

    @abstractmethod
    async def taskqueue_get_next_task(self, project_id: str) -> dict[str, Any]:
        """Get the next task to be done.

        Args:
            project_id: Project ID

        Returns:
            Response dictionary with task or None
        """
        pass

    # Temporal operations
    @abstractmethod
    async def temporal_get_workflow_history(
        self,
        workflow_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Get workflow execution history.

        Args:
            workflow_id: Workflow ID
            run_id: Optional run ID

        Returns:
            Response dictionary with workflow history or error
        """
        pass


class RealMCPClient(MCPClient):
    """Real MCP client using Claude Code injected functions.

    This implementation calls actual MCP server functions that are injected
    into the global namespace by Claude Code at runtime.
    """

    # Redis operations
    async def redis_set(
        self,
        key: str,
        value: Any,
        expiration: int | None = None,
    ) -> dict[str, Any]:
        """Set a Redis key with optional expiration."""
        if "mcp__RedisMCPServer__set" in globals():
            func = globals()["mcp__RedisMCPServer__set"]
            return func(key=key, value=value, expiration=expiration)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_get(self, key: str) -> dict[str, Any]:
        """Get a Redis key value."""
        if "mcp__RedisMCPServer__get" in globals():
            func = globals()["mcp__RedisMCPServer__get"]
            return func(key=key)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_delete(self, key: str) -> dict[str, Any]:
        """Delete a Redis key."""
        if "mcp__RedisMCPServer__delete" in globals():
            func = globals()["mcp__RedisMCPServer__delete"]
            return func(key=key)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_hset(
        self,
        name: str,
        key: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Set a hash field with optional expiration."""
        if "mcp__RedisMCPServer__hset" in globals():
            func = globals()["mcp__RedisMCPServer__hset"]
            return func(
                name=name,
                key=key,
                value=value,
                expire_seconds=expire_seconds,
            )
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_hget(self, name: str, key: str) -> dict[str, Any]:
        """Get a hash field value."""
        if "mcp__RedisMCPServer__hget" in globals():
            func = globals()["mcp__RedisMCPServer__hget"]
            return func(name=name, key=key)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_hgetall(self, name: str) -> dict[str, Any]:
        """Get all fields and values from a hash."""
        if "mcp__RedisMCPServer__hgetall" in globals():
            func = globals()["mcp__RedisMCPServer__hgetall"]
            return func(name=name)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_lpush(
        self,
        name: str,
        value: Any,
        expire: int | None = None,
    ) -> dict[str, Any]:
        """Push a value to a Redis list."""
        if "mcp__RedisMCPServer__lpush" in globals():
            func = globals()["mcp__RedisMCPServer__lpush"]
            return func(name=name, value=value, expire=expire)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_llen(self, name: str) -> dict[str, Any]:
        """Get the length of a Redis list."""
        if "mcp__RedisMCPServer__llen" in globals():
            func = globals()["mcp__RedisMCPServer__llen"]
            return func(name=name)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_json_set(
        self,
        name: str,
        path: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Set a JSON value at path."""
        if "mcp__RedisMCPServer__json_set" in globals():
            func = globals()["mcp__RedisMCPServer__json_set"]
            return func(
                name=name,
                path=path,
                value=value,
                expire_seconds=expire_seconds,
            )
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_json_get(
        self,
        name: str,
        path: str = "$",
    ) -> dict[str, Any]:
        """Get a JSON value at path."""
        if "mcp__RedisMCPServer__json_get" in globals():
            func = globals()["mcp__RedisMCPServer__json_get"]
            return func(name=name, path=path)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_publish(self, channel: str, message: str) -> dict[str, Any]:
        """Publish a message to a Redis channel."""
        if "mcp__RedisMCPServer__publish" in globals():
            func = globals()["mcp__RedisMCPServer__publish"]
            return func(channel=channel, message=message)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def redis_subscribe(self, channel: str) -> dict[str, Any]:
        """Subscribe to a Redis channel."""
        if "mcp__RedisMCPServer__subscribe" in globals():
            func = globals()["mcp__RedisMCPServer__subscribe"]
            return func(channel=channel)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    # TaskQueue operations
    async def taskqueue_create_project(
        self,
        initial_prompt: str,
        tasks: list[dict[str, str]],
        auto_approve: bool = False,
    ) -> dict[str, Any]:
        """Create a new project with tasks."""
        if "mcp__taskqueue__create_project" in globals():
            func = globals()["mcp__taskqueue__create_project"]
            return func(
                initialPrompt=initial_prompt,
                tasks=tasks,
                autoApprove=auto_approve,
            )
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def taskqueue_list_projects(self, state: str = "open") -> dict[str, Any]:
        """List projects by state."""
        if "mcp__taskqueue__list_projects" in globals():
            func = globals()["mcp__taskqueue__list_projects"]
            return func(state=state)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def taskqueue_read_project(self, project_id: str) -> dict[str, Any]:
        """Read project details."""
        if "mcp__taskqueue__read_project" in globals():
            func = globals()["mcp__taskqueue__read_project"]
            return func(projectId=project_id)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def taskqueue_add_tasks_to_project(
        self,
        project_id: str,
        tasks: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Add tasks to an existing project."""
        if "mcp__taskqueue__add_tasks_to_project" in globals():
            func = globals()["mcp__taskqueue__add_tasks_to_project"]
            return func(
                projectId=project_id,
                tasks=tasks,
            )
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def taskqueue_update_task(
        self,
        project_id: str,
        task_id: str,
        status: str,
        completed_details: str | None = None,
    ) -> dict[str, Any]:
        """Update task status."""
        if "mcp__taskqueue__update_task" in globals():
            func = globals()["mcp__taskqueue__update_task"]
            return func(
                projectId=project_id,
                taskId=task_id,
                status=status,
                completedDetails=completed_details,
            )
        raise RuntimeError("MCP function not available - must run via Claude Code")

    async def taskqueue_get_next_task(self, project_id: str) -> dict[str, Any]:
        """Get the next task to be done."""
        if "mcp__taskqueue__get_next_task" in globals():
            func = globals()["mcp__taskqueue__get_next_task"]
            return func(projectId=project_id)
        raise RuntimeError("MCP function not available - must run via Claude Code")

    # Temporal operations
    async def temporal_get_workflow_history(
        self,
        workflow_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Get workflow execution history."""
        if "mcp__temporal_mcp__GetWorkflowHistory" in globals():
            func = globals()["mcp__temporal_mcp__GetWorkflowHistory"]
            return func(
                workflowId=workflow_id,
                runId=run_id,
            )
        raise RuntimeError("MCP function not available - must run via Claude Code")


class MockMCPClient(MCPClient):
    """Mock MCP client for testing.

    This implementation uses existing mock classes from tests/fixtures/mocks.py
    to provide test-friendly responses without requiring real MCP servers.
    """

    def __init__(self) -> None:
        """Initialize mock MCP client with mock server instances."""
        from tests.fixtures.mocks import MockRedisMCP, MockTaskQueueMCP, MockTemporalMCP

        self.redis = MockRedisMCP()
        self.temporal = MockTemporalMCP()
        self.taskqueue = MockTaskQueueMCP()

    # Redis operations
    async def redis_set(
        self,
        key: str,
        value: Any,
        expiration: int | None = None,
    ) -> dict[str, Any]:
        """Set a Redis key with optional expiration."""
        return await self.redis.set(key=key, value=value, expiration=expiration)

    async def redis_get(self, key: str) -> dict[str, Any]:
        """Get a Redis key value."""
        return await self.redis.get(key=key)

    async def redis_delete(self, key: str) -> dict[str, Any]:
        """Delete a Redis key."""
        return await self.redis.delete(key=key)

    async def redis_hset(
        self,
        name: str,
        key: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Set a hash field with optional expiration."""
        return await self.redis.hset(
            name=name,
            key=key,
            value=value,
            expire_seconds=expire_seconds,
        )

    async def redis_hget(self, name: str, key: str) -> dict[str, Any]:
        """Get a hash field value."""
        return await self.redis.hget(name=name, key=key)

    async def redis_hgetall(self, name: str) -> dict[str, Any]:
        """Get all fields and values from a hash."""
        return await self.redis.hgetall(name=name)

    async def redis_lpush(
        self,
        name: str,
        value: Any,
        expire: int | None = None,
    ) -> dict[str, Any]:
        """Push a value to a Redis list."""
        return await self.redis.lpush(name=name, value=value, expire=expire)

    async def redis_llen(self, name: str) -> dict[str, Any]:
        """Get the length of a Redis list."""
        return await self.redis.llen(name=name)

    async def redis_json_set(
        self,
        name: str,
        path: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Set a JSON value at path."""
        return await self.redis.json_set(
            name=name,
            path=path,
            value=value,
            expire_seconds=expire_seconds,
        )

    async def redis_json_get(
        self,
        name: str,
        path: str = "$",
    ) -> dict[str, Any]:
        """Get a JSON value at path."""
        return await self.redis.json_get(name=name, path=path)

    async def redis_publish(self, channel: str, message: str) -> dict[str, Any]:
        """Publish a message to a Redis channel."""
        return await self.redis.publish(channel=channel, message=message)

    async def redis_subscribe(self, channel: str) -> dict[str, Any]:
        """Subscribe to a Redis channel."""
        return await self.redis.subscribe(channel=channel)

    # TaskQueue operations
    async def taskqueue_create_project(
        self,
        initial_prompt: str,
        tasks: list[dict[str, str]],
        auto_approve: bool = False,
    ) -> dict[str, Any]:
        """Create a new project with tasks."""
        return await self.taskqueue.create_project(
            initial_prompt=initial_prompt,
            tasks=tasks,
            auto_approve=auto_approve,
        )

    async def taskqueue_list_projects(self, state: str = "open") -> dict[str, Any]:
        """List projects by state."""
        return await self.taskqueue.list_projects(state=state)

    async def taskqueue_read_project(self, project_id: str) -> dict[str, Any]:
        """Read project details."""
        return await self.taskqueue.read_project(project_id=project_id)

    async def taskqueue_add_tasks_to_project(
        self,
        project_id: str,
        tasks: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Add tasks to an existing project."""
        return await self.taskqueue.add_tasks_to_project(
            project_id=project_id,
            tasks=tasks,
        )

    async def taskqueue_update_task(
        self,
        project_id: str,
        task_id: str,
        status: str,
        completed_details: str | None = None,
    ) -> dict[str, Any]:
        """Update task status."""
        return await self.taskqueue.update_task(
            project_id=project_id,
            task_id=task_id,
            status=status,
            completed_details=completed_details,
        )

    async def taskqueue_get_next_task(self, project_id: str) -> dict[str, Any]:
        """Get the next task to be done."""
        return await self.taskqueue.get_next_task(project_id=project_id)

    # Temporal operations
    async def temporal_get_workflow_history(
        self,
        workflow_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Get workflow execution history."""
        return await self.temporal.get_workflow_history(
            workflow_id=workflow_id,
            run_id=run_id,
        )


# Factory function for easy client creation
def create_mcp_client(use_mock: bool = False) -> MCPClient:
    """Create an MCP client instance.

    Args:
        use_mock: If True, create a mock client. Otherwise create a real client.

    Returns:
        MCPClient instance (either RealMCPClient or MockMCPClient)

    Example:
        # In production code (Claude Code environment)
        client = create_mcp_client(use_mock=False)

        # In tests
        client = create_mcp_client(use_mock=True)
    """
    if use_mock:
        return MockMCPClient()
    return RealMCPClient()

"""Convenience wrappers for common MCP operations.

This module provides Python-friendly wrappers around MCP server tools
for Redis, Temporal, and TaskQueue operations.
"""

from typing import Any


class RedisWrapper:
    """Wrapper for RedisMCPServer operations."""

    @staticmethod
    async def set_key(key: str, value: Any, expire_seconds: int | None = None) -> bool:
        """Set a Redis key with optional expiration.

        Args:
            key: Redis key
            value: Value to store
            expire_seconds: Optional expiration time in seconds

        Returns:
            True if successful, False otherwise

        Note:
            Uses mcp__RedisMCPServer__set tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("Redis set to be implemented")

    @staticmethod
    async def get_key(key: str) -> Any | None:
        """Get a Redis key value.

        Args:
            key: Redis key

        Returns:
            Value if key exists, None otherwise

        Note:
            Uses mcp__RedisMCPServer__get tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("Redis get to be implemented")

    @staticmethod
    async def publish(channel: str, message: str) -> int:
        """Publish a message to a Redis channel.

        Args:
            channel: Channel name
            message: Message content

        Returns:
            Number of subscribers that received the message

        Note:
            Uses mcp__RedisMCPServer__publish tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("Redis publish to be implemented")

    @staticmethod
    async def subscribe(channel: str) -> None:
        """Subscribe to a Redis channel.

        Args:
            channel: Channel name

        Note:
            Uses mcp__RedisMCPServer__subscribe tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("Redis subscribe to be implemented")

    @staticmethod
    async def hset(
        name: str,
        key: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> bool:
        """Set a hash field with optional expiration.

        Args:
            name: Hash name
            key: Field name
            value: Field value
            expire_seconds: Optional expiration time in seconds

        Returns:
            True if successful, False otherwise

        Note:
            Uses mcp__RedisMCPServer__hset tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("Redis hset to be implemented")

    @staticmethod
    async def hgetall(name: str) -> dict[str, Any]:
        """Get all fields and values from a hash.

        Args:
            name: Hash name

        Returns:
            Dictionary of field-value pairs

        Note:
            Uses mcp__RedisMCPServer__hgetall tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("Redis hgetall to be implemented")


class TemporalWrapper:
    """Wrapper for temporal-mcp operations."""

    @staticmethod
    async def get_workflow_history(
        workflow_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Get workflow execution history.

        Args:
            workflow_id: Workflow ID
            run_id: Optional run ID (uses latest if not provided)

        Returns:
            Workflow history data

        Note:
            Uses mcp__temporal-mcp__GetWorkflowHistory tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("Temporal workflow history to be implemented")


class TaskQueueWrapper:
    """Wrapper for taskqueue MCP operations."""

    @staticmethod
    async def create_project(
        initial_prompt: str,
        tasks: list[dict[str, str]],
        auto_approve: bool = False,
    ) -> str:
        """Create a new project with tasks.

        Args:
            initial_prompt: Initial project prompt
            tasks: List of task dictionaries with 'title' and 'description'
            auto_approve: Whether to auto-approve completed tasks

        Returns:
            Project ID

        Note:
            Uses mcp__taskqueue__create_project tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("TaskQueue create_project to be implemented")

    @staticmethod
    async def list_projects(state: str = "open") -> list[dict[str, Any]]:
        """List projects by state.

        Args:
            state: Project state filter (open, pending_approval, completed, all)

        Returns:
            List of project information

        Note:
            Uses mcp__taskqueue__list_projects tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("TaskQueue list_projects to be implemented")

    @staticmethod
    async def add_tasks(
        project_id: str,
        tasks: list[dict[str, str]],
    ) -> None:
        """Add tasks to an existing project.

        Args:
            project_id: Project ID
            tasks: List of task dictionaries with 'title' and 'description'

        Note:
            Uses mcp__taskqueue__add_tasks_to_project tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("TaskQueue add_tasks to be implemented")

    @staticmethod
    async def update_task(
        project_id: str,
        task_id: str,
        status: str,
        completed_details: str | None = None,
    ) -> None:
        """Update task status.

        Args:
            project_id: Project ID
            task_id: Task ID
            status: New status (not started, in progress, done)
            completed_details: Required if status is 'done'

        Note:
            Uses mcp__taskqueue__update_task tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("TaskQueue update_task to be implemented")

    @staticmethod
    async def get_next_task(project_id: str) -> dict[str, Any] | None:
        """Get the next task to be done in a project.

        Args:
            project_id: Project ID

        Returns:
            Next task information or None if no tasks pending

        Note:
            Uses mcp__taskqueue__get_next_task tool
        """
        # TODO: Implement via MCP tool call
        raise NotImplementedError("TaskQueue get_next_task to be implemented")

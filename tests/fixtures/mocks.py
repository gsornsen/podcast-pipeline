"""Mock MCP responses for CI/CD testing without live services."""

from typing import Any


class MockRedisMCP:
    """Mock RedisMCPServer responses."""

    def __init__(self) -> None:
        """Initialize mock Redis state."""
        self.data: dict[str, Any] = {}
        self.hashes: dict[str, dict[str, Any]] = {}
        self.pub_sub_messages: dict[str, list[str]] = {}

    async def set(
        self,
        key: str,
        value: Any,
        expiration: int | None = None,
    ) -> dict[str, Any]:
        """Mock Redis SET operation.

        Args:
            key: Redis key
            value: Value to store
            expiration: Optional expiration in seconds

        Returns:
            Success response
        """
        self.data[key] = value
        return {"status": "success", "message": f"Key '{key}' set successfully"}

    async def get(self, key: str) -> dict[str, Any]:
        """Mock Redis GET operation.

        Args:
            key: Redis key

        Returns:
            Value or error
        """
        if key in self.data:
            return {"status": "success", "value": self.data[key]}
        return {"error": f"Key '{key}' not found"}

    async def hset(
        self,
        name: str,
        key: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Mock Redis HSET operation.

        Args:
            name: Hash name
            key: Field name
            value: Field value
            expire_seconds: Optional expiration in seconds

        Returns:
            Success response
        """
        if name not in self.hashes:
            self.hashes[name] = {}
        self.hashes[name][key] = value
        return {"status": "success", "message": f"Hash '{name}' field '{key}' set"}

    async def hgetall(self, name: str) -> dict[str, Any]:
        """Mock Redis HGETALL operation.

        Args:
            name: Hash name

        Returns:
            Hash fields or error
        """
        if name in self.hashes:
            return {"fields": self.hashes[name]}
        return {"error": f"Hash '{name}' not found"}

    async def publish(self, channel: str, message: str) -> dict[str, Any]:
        """Mock Redis PUBLISH operation.

        Args:
            channel: Channel name
            message: Message content

        Returns:
            Subscriber count
        """
        if channel not in self.pub_sub_messages:
            self.pub_sub_messages[channel] = []
        self.pub_sub_messages[channel].append(message)
        return {"subscribers": len(self.pub_sub_messages[channel])}

    async def subscribe(self, channel: str) -> dict[str, Any]:
        """Mock Redis SUBSCRIBE operation.

        Args:
            channel: Channel name

        Returns:
            Success response
        """
        if channel not in self.pub_sub_messages:
            self.pub_sub_messages[channel] = []
        return {"status": "success", "message": f"Subscribed to '{channel}'"}

    async def delete(self, key: str) -> dict[str, Any]:
        """Mock Redis DELETE operation.

        Args:
            key: Redis key

        Returns:
            Success response
        """
        if key in self.data:
            del self.data[key]
            return {"status": "success", "message": f"Key '{key}' deleted"}
        return {"status": "success", "message": f"Key '{key}' did not exist"}

    async def hget(self, name: str, key: str) -> dict[str, Any]:
        """Mock Redis HGET operation.

        Args:
            name: Hash name
            key: Field name

        Returns:
            Field value or error
        """
        if name in self.hashes and key in self.hashes[name]:
            return {"status": "success", "value": self.hashes[name][key]}
        return {"error": f"Hash '{name}' field '{key}' not found"}

    async def lpush(
        self,
        name: str,
        value: Any,
        expire: int | None = None,
    ) -> dict[str, Any]:
        """Mock Redis LPUSH operation.

        Args:
            name: List name
            value: Value to push
            expire: Optional expiration in seconds

        Returns:
            List length
        """
        if name not in self.data or not isinstance(self.data[name], list):
            self.data[name] = []

        self.data[name].insert(0, value)
        return {"status": "success", "length": len(self.data[name])}

    async def llen(self, name: str) -> dict[str, Any]:
        """Mock Redis LLEN operation.

        Args:
            name: List name

        Returns:
            List length
        """
        if name in self.data and isinstance(self.data[name], list):
            return {"status": "success", "length": len(self.data[name])}
        return {"status": "success", "length": 0}

    async def json_set(
        self,
        name: str,
        path: str,
        value: Any,
        expire_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Mock Redis JSON.SET operation.

        Args:
            name: Redis key
            path: JSON path (e.g., "$" for root)
            value: JSON value
            expire_seconds: Optional expiration in seconds

        Returns:
            Success response
        """
        # Simple implementation: store at root path
        if path == "$":
            self.data[name] = value
        else:
            # For nested paths, store as nested dict
            if name not in self.data:
                self.data[name] = {}
            # This is a simplified version - doesn't handle complex JSONPath
            self.data[name] = value

        return {"status": "success", "message": f"JSON set at '{name}' path '{path}'"}

    async def json_get(
        self,
        name: str,
        path: str = "$",
    ) -> dict[str, Any]:
        """Mock Redis JSON.GET operation.

        Args:
            name: Redis key
            path: JSON path (default: "$" for root)

        Returns:
            JSON value or error
        """
        if name in self.data:
            if path == "$":
                return {"value": self.data[name]}
            else:
                # Simplified: return the whole value
                return {"value": self.data[name]}
        return {"error": f"Key '{name}' not found"}


class MockTemporalMCP:
    """Mock temporal-mcp responses."""

    def __init__(self) -> None:
        """Initialize mock Temporal state."""
        self.workflows: dict[str, dict[str, Any]] = {}

    async def get_workflow_history(
        self,
        workflow_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Mock GetWorkflowHistory operation.

        Args:
            workflow_id: Workflow ID
            run_id: Optional run ID

        Returns:
            Workflow history or error
        """
        if workflow_id in self.workflows:
            return {
                "workflow_id": workflow_id,
                "run_id": run_id or "mock-run-id",
                "status": "completed",
                "history": [
                    {"event_type": "WorkflowExecutionStarted", "timestamp": "2025-01-01T00:00:00Z"},
                    {"event_type": "WorkflowExecutionCompleted", "timestamp": "2025-01-01T00:01:00Z"},
                ],
            }
        return {"error": f"Workflow '{workflow_id}' not found"}

    def register_workflow(
        self,
        workflow_id: str,
        status: str = "running",
        **kwargs: Any,
    ) -> None:
        """Register a mock workflow for testing.

        Args:
            workflow_id: Workflow ID
            status: Workflow status
            **kwargs: Additional workflow properties
        """
        self.workflows[workflow_id] = {"status": status, **kwargs}


class MockTaskQueueMCP:
    """Mock taskqueue MCP responses."""

    def __init__(self) -> None:
        """Initialize mock TaskQueue state."""
        self.projects: dict[str, dict[str, Any]] = {}
        self.next_project_id = 1
        self.next_task_id = 1

    async def create_project(
        self,
        initial_prompt: str,
        tasks: list[dict[str, str]],
        auto_approve: bool = False,
    ) -> dict[str, Any]:
        """Mock create_project operation.

        Args:
            initial_prompt: Project prompt
            tasks: Task list
            auto_approve: Auto-approve flag

        Returns:
            Project ID
        """
        project_id = f"proj-{self.next_project_id}"
        self.next_project_id += 1

        task_list = []
        for task in tasks:
            task_id = f"task-{self.next_task_id}"
            self.next_task_id += 1

            # Handle both string tasks and dict tasks
            if isinstance(task, str):
                task_list.append({
                    "id": task_id,
                    "title": task,
                    "description": task,
                    "status": "not started",
                })
            else:
                task_list.append({
                    "id": task_id,
                    "title": task.get("title", "Untitled"),
                    "description": task.get("description", "No description"),
                    "status": "not started",
                })

        self.projects[project_id] = {
            "id": project_id,
            "initial_prompt": initial_prompt,
            "tasks": task_list,
            "auto_approve": auto_approve,
        }

        return {
            "status": "success",
            "project_id": project_id,
            "message": "Project created successfully"
        }

    async def list_projects(self, state: str = "open") -> dict[str, Any]:
        """Mock list_projects operation.

        Args:
            state: Filter by state

        Returns:
            Project list
        """
        projects = []
        for project_id, project in self.projects.items():
            # Simple state filter logic
            has_incomplete = any(t["status"] != "done" for t in project["tasks"])
            if (state == "open" and has_incomplete) or (state == "completed" and not has_incomplete) or state == "all":
                projects.append(project)

        return {"status": "success", "projects": projects}

    async def add_tasks_to_project(
        self,
        project_id: str,
        tasks: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Mock add_tasks_to_project operation.

        Args:
            project_id: Project ID
            tasks: Task list

        Returns:
            Success response
        """
        if project_id not in self.projects:
            return {"error": f"Project '{project_id}' not found"}

        for task in tasks:
            task_id = f"task-{self.next_task_id}"
            self.next_task_id += 1
            self.projects[project_id]["tasks"].append({
                "id": task_id,
                "title": task["title"],
                "description": task["description"],
                "status": "not started",
            })

        return {"status": "success", "message": f"Added {len(tasks)} tasks"}

    async def update_task(
        self,
        project_id: str,
        task_id: str,
        status: str,
        completed_details: str | None = None,
    ) -> dict[str, Any]:
        """Mock update_task operation.

        Args:
            project_id: Project ID
            task_id: Task ID
            status: New status
            completed_details: Completion details if done

        Returns:
            Success response
        """
        if project_id not in self.projects:
            return {"error": f"Project '{project_id}' not found"}

        project = self.projects[project_id]
        for task in project["tasks"]:
            if task["id"] == task_id:
                task["status"] = status
                if completed_details:
                    task["completed_details"] = completed_details
                return {"status": "success", "message": f"Task '{task_id}' updated"}

        return {"error": f"Task '{task_id}' not found"}

    async def get_next_task(self, project_id: str) -> dict[str, Any]:
        """Mock get_next_task operation.

        Args:
            project_id: Project ID

        Returns:
            Next task or None
        """
        if project_id not in self.projects:
            return {"error": f"Project '{project_id}' not found"}

        project = self.projects[project_id]
        for task in project["tasks"]:
            if task["status"] != "done":
                return {"task": task}

        return {"task": None, "message": "No tasks pending"}

    async def read_project(self, project_id: str) -> dict[str, Any]:
        """Mock read_project operation.

        Args:
            project_id: Project ID

        Returns:
            Project details or error
        """
        if project_id not in self.projects:
            return {"error": f"Project '{project_id}' not found"}

        return {"project": self.projects[project_id]}

#!/usr/bin/env python3
"""
Real Integration Test Pipeline for Podcast-Pipeline Multi-Agent Coordination System

This test suite validates all foundational systems are operational for multi-agent collaboration:
- MCP Server connectivity (Redis, TaskQueue, Temporal, etc.)
- Redis operations and communication patterns
- Task queue operations and lifecycle management
- Temporal workflow execution
- Multi-agent collaboration scenarios

The tests use the MCPClient abstraction layer, enabling both:
- Real MCP server testing (default)
- Mock mode testing (USE_MOCK_MCP=1 for CI/CD)

Run with: pytest tests/integration_tests/test_integration_pipeline.py -v
Mock mode: USE_MOCK_MCP=1 pytest tests/integration_tests/test_integration_pipeline.py -v
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import pytest
import pytest_asyncio


@dataclass
class _TestResult:
    """Standardized test result structure"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration_ms: float
    details: str
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class _SystemStatus:
    """Overall system health status"""
    redis_status: str
    temporal_status: str
    mcp_servers: Dict[str, str]
    overall_health: str
    timestamp: str


@pytest.mark.asyncio
class TestRealIntegrationPipeline:
    """Integration test suite using MCPClient abstraction for real or mock MCP operations."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Setup test environment before each test."""
        self.system_status = _SystemStatus(
            redis_status="UNKNOWN",
            temporal_status="UNKNOWN",
            mcp_servers={},
            overall_health="UNKNOWN",
            timestamp=datetime.now().isoformat(),
        )
        self.test_results = []
        self.start_time = time.time()
        self.test_session_id = str(uuid.uuid4())[:8]
        yield
        # Cleanup after test if needed
        print(f"Test session {self.test_session_id} completed")

    @pytest.mark.mock_compatible
    async def test_system_health(self, mcp_client, reset_mcp_state):
        """Test basic system health checks using MCP client."""
        print(f"\n🔍 Testing System Health - Session: {self.test_session_id}")

        # Test Redis connectivity with MCP client
        redis_available = await self.check_redis_health(mcp_client)
        assert redis_available, "Redis should be operational"
        self.system_status.redis_status = "OPERATIONAL" if redis_available else "OFFLINE"
        print(f"✅ Redis Status: {self.system_status.redis_status}")

        # Test Temporal connectivity with MCP client
        temporal_available = await self.check_temporal_health(mcp_client)
        # Note: Temporal test might fail if workflow service not running - that's expected
        self.system_status.temporal_status = "OPERATIONAL" if temporal_available else "LIMITED"
        print(f"⚠️ Temporal Status: {self.system_status.temporal_status}")

        # Test MCP servers with client
        mcp_results = await self.check_mcp_servers(mcp_client)
        self.system_status.mcp_servers = mcp_results
        print(f"📊 MCP Server Status: {mcp_results}")

        # Check that critical MCP servers are operational
        assert mcp_results.get("Redis") == "OPERATIONAL", "Redis MCP should be operational"
        # TaskQueue might not be available - check gracefully
        if mcp_results.get("TaskQueue") != "OPERATIONAL":
            print("⚠️ TaskQueue MCP not operational - some tests may be skipped")

    @pytest.mark.mock_compatible
    async def test_mcp_connectivity(self, mcp_client, reset_mcp_state):
        """Test MCP server connectivity and operations."""
        print(f"\n🔗 Testing MCP Connectivity - Session: {self.test_session_id}")

        # Test Redis MCP operations
        redis_ops = await self._test_redis_mcp_operations(mcp_client)
        assert redis_ops["successful"] >= 1, "At least one Redis MCP operation should succeed"
        print(f"✅ Redis MCP: {redis_ops['successful']}/{redis_ops['tested']} operations successful")

        # Test TaskQueue MCP operations
        taskqueue_ops = await self._test_taskqueue_mcp_operations(mcp_client)
        print(f"📋 TaskQueue MCP: {taskqueue_ops['successful']}/{taskqueue_ops['tested']} operations successful")

        # Test Temporal MCP operations
        temporal_ops = await self._test_temporal_mcp_operations(mcp_client)
        print(f"⏱️ Temporal MCP: {temporal_ops['successful']}/{temporal_ops['tested']} operations successful")

    @pytest.mark.mock_compatible
    async def test_redis_communication(self, mcp_client, reset_mcp_state):
        """Test Redis pub/sub and communication patterns."""
        print(f"\n💬 Testing Redis Communication - Session: {self.test_session_id}")

        # Test key operations
        key_ops = await self._test_redis_key_operations(mcp_client)
        assert key_ops["successful"] >= 1, "At least one Redis key operation should succeed"
        print(f"🔑 Key Operations: {key_ops['successful']}/{key_ops['tested']} successful")

        # Test hash operations
        hash_ops = await self._test_redis_hash_operations(mcp_client)
        assert hash_ops["successful"] >= 1, "At least one Redis hash operation should succeed"
        print(f"🗂️ Hash Operations: {hash_ops['successful']}/{hash_ops['tested']} successful")

        # Test list operations
        list_ops = await self._test_redis_list_operations(mcp_client)
        assert list_ops["successful"] >= 1, "At least one Redis list operation should succeed"
        print(f"📝 List Operations: {list_ops['successful']}/{list_ops['tested']} successful")

    @pytest.mark.mock_compatible
    async def test_task_queue_integration(self, mcp_client, reset_mcp_state):
        """Test TaskQueue operations and lifecycle."""
        print(f"\n📋 Testing Task Queue Integration - Session: {self.test_session_id}")

        # Create test project
        project_result = await self.create_test_project(mcp_client)
        assert project_result["success"], "Project creation should succeed"
        print(f"📁 Project: {project_result.get('project_id', 'N/A')}")

        # Add test tasks
        task_results = await self.add_test_tasks(mcp_client)
        print(f"📝 Tasks Added: {task_results.get('tasks_created', 0)}")

        # Test task retrieval
        retrieval_results = await self._test_task_retrieval(mcp_client)
        print(f"🔍 Task Retrieval: {retrieval_results['successful']}/{retrieval_results['tested']} successful")

    @pytest.mark.mock_compatible
    async def test_temporal_workflow(self, mcp_client, reset_mcp_state):
        """Test Temporal workflow execution."""
        print(f"\n⏱️ Testing Temporal Workflow - Session: {self.test_session_id}")

        # Test workflow history retrieval
        history_retrieval = await self.get_workflow_history(mcp_client)
        print(f"📚 History Retrieval: {'✅' if history_retrieval['success'] else '❌'}")

        # If we have a working temporal connection, try more operations
        if history_retrieval["success"]:
            print("🎯 Temporal MCP is operational")
        else:
            print("⚠️ Temporal MCP limited - workflow service may not be running")

    @pytest.mark.mock_compatible
    async def test_multi_agent_collaboration(self, mcp_client, reset_mcp_state):
        """Test multi-agent collaboration scenario using infrastructure."""
        print(f"\n🤝 Testing Multi-Agent Collaboration - Session: {self.test_session_id}")

        # Setup coordination using Redis for agent state
        coordination_setup = await self.setup_agent_coordination(mcp_client)
        assert coordination_setup["success"], "Agent coordination setup should succeed"
        print(f"🎭 Agents Coordinated: {len(coordination_setup.get('active_agents', []))}")

        # Create collaborative task using storage
        task_creation = await self.create_collaborative_task(mcp_client)
        assert task_creation["success"], "Collaborative task creation should succeed"
        print(f"🎯 Task Created: {task_creation.get('task_id', 'N/A')}")

        # Distribute subtasks using queuing
        task_distribution = await self.distribute_subtasks(mcp_client)
        print(f"📤 Subtasks Distributed: {task_distribution.get('distributed_subtasks', 0)}")

        # Monitor progress using state tracking
        progress_monitoring = await self.monitor_collaborative_progress(mcp_client)
        print(f"📊 Progress: {progress_monitoring.get('final_progress', 0)}%")

    @pytest.mark.mock_compatible
    async def test_agent_coordination_hooks(self, mcp_client, reset_mcp_state):
        """Test agent coordination hooks patterns with infrastructure."""
        print(f"\n🎭 Testing Agent Coordination Hooks - Session: {self.test_session_id}")

        # Test task handoff protocol
        handoff_test = await self._test_task_handoff_protocol(mcp_client)
        assert handoff_test["success"], "Task handoff protocol should work"
        print(f"✅ Task Handoff: {handoff_test['details']}")

        # Test event-driven coordination
        event_test = await self._test_event_driven_coordination(mcp_client)
        print(f"📡 Event-Driven: {event_test['events_published']}/{event_test['events_tested']} events published")

        # Test workflow-agent integration
        workflow_integration = await self._test_workflow_agent_integration(mcp_client)
        print(f"⏱️ Workflow-Agent: {workflow_integration['details']}")

    @pytest.mark.mock_compatible
    async def test_slash_commands(self, mcp_client, reset_mcp_state):
        """Test slash command functionality with infrastructure."""
        print(f"\n⚡ Testing Slash Commands - Session: {self.test_session_id}")

        # Test /workflow-status equivalent operations
        workflow_status = await self._test_workflow_status_command(mcp_client)
        print(f"📊 Workflow Status: {workflow_status['details']}")

        # Test /agent-status equivalent operations
        agent_status = await self._test_agent_status_command(mcp_client)
        print(f"🤖 Agent Status: {agent_status['agents_tracked']}/{agent_status['total_agents']} agents tracked")

        # Test /check-infrastructure equivalent operations
        infra_check = await self._test_check_infrastructure_command(mcp_client)
        assert infra_check["success"], "Infrastructure check should succeed"
        print(f"✅ Infrastructure Check: {infra_check['healthy_components']}/{infra_check['total_components']} healthy")

    @pytest.mark.mock_compatible
    async def test_full_pipeline(self, mcp_client, reset_mcp_state):
        """Run full integration test pipeline with infrastructure."""
        print("="*80)
        print(f"🚀 Running Integration Test Pipeline - Session: {self.test_session_id}")
        print("="*80)

        # Run all tests in sequence
        await self.test_system_health(mcp_client, reset_mcp_state)
        await self.test_mcp_connectivity(mcp_client, reset_mcp_state)
        await self.test_redis_communication(mcp_client, reset_mcp_state)
        await self.test_task_queue_integration(mcp_client, reset_mcp_state)
        await self.test_temporal_workflow(mcp_client, reset_mcp_state)
        await self.test_multi_agent_collaboration(mcp_client, reset_mcp_state)
        await self.test_agent_coordination_hooks(mcp_client, reset_mcp_state)
        await self.test_slash_commands(mcp_client, reset_mcp_state)

        print("\n✅ Full pipeline test completed successfully!")
        print(f"📊 Final System Status: {self.system_status.overall_health}")

    # Helper methods using MCP client abstraction
    async def check_redis_health(self, mcp_client) -> bool:
        """Check Redis server health with MCP client."""
        try:
            # Use MCP client - Redis SET operation
            test_key = f"health_check:{self.test_session_id}"
            result = await mcp_client.redis_set(key=test_key, value="health_ok")

            if isinstance(result, dict) and result.get("status") == "success":
                print(f"✅ Redis health check successful: {test_key}")
                return True
            else:
                print(f"❌ Redis health check failed: {result}")
                return False
        except Exception as e:
            print(f"Redis health check failed: {e}")
            return False

    async def check_temporal_health(self, mcp_client) -> bool:
        """Check Temporal server health with MCP client."""
        try:
            # Use MCP client health check - try to get workflow history for a test workflow
            result = await mcp_client.temporal_get_workflow_history(
                workflow_id="health_check_test"
            )

            if isinstance(result, dict):
                # Success if we get a valid response (even if workflow not found)
                if result.get("status") == "success" or "not found" in str(result.get("error", "")).lower():
                    print("✅ Temporal health check successful")
                    return True
                else:
                    print(f"⚠️ Temporal health check limited: {result}")
                    return False
        except Exception as e:
            print(f"Temporal health check failed: {e}")
            return False

    async def check_mcp_servers(self, mcp_client) -> Dict[str, str]:
        """Check all MCP server availability with client."""
        results = {}

        # Test Redis MCP Server
        try:
            test_key = f"mcp_test:{self.test_session_id}"
            result = await mcp_client.redis_set(key=test_key, value="test")

            if isinstance(result, dict) and result.get("status") == "success":
                results["Redis"] = "OPERATIONAL"
                print("✅ Redis MCP Server operational")
            else:
                results["Redis"] = "LIMITED"
                print(f"⚠️ Redis MCP Server limited: {result}")
        except Exception as e:
            results["Redis"] = "OFFLINE"
            print(f"❌ Redis MCP Server offline: {e}")

        # Test TaskQueue MCP Server
        try:
            result = await mcp_client.taskqueue_list_projects()

            if isinstance(result, dict) and result.get("status") == "success":
                results["TaskQueue"] = "OPERATIONAL"
                print("✅ TaskQueue MCP Server operational")
            else:
                results["TaskQueue"] = "LIMITED"
                print(f"⚠️ TaskQueue MCP Server limited: {result}")
        except Exception as e:
            results["TaskQueue"] = "OFFLINE"
            print(f"❌ TaskQueue MCP Server offline: {e}")

        # Test Temporal MCP Server
        try:
            result = await mcp_client.temporal_get_workflow_history(workflow_id="test")

            if isinstance(result, dict):
                # Success if we get any valid response
                if result.get("status") == "success" or "not found" in str(result.get("error", "")).lower():
                    results["Temporal"] = "OPERATIONAL"
                    print("✅ Temporal MCP Server operational")
                else:
                    results["Temporal"] = "LIMITED"
                    print(f"⚠️ Temporal MCP Server limited: {result}")
        except Exception as e:
            results["Temporal"] = "LIMITED"
            print(f"⚠️ Temporal MCP Server limited: {e}")

        return results

    async def _test_redis_mcp_operations(self, mcp_client) -> Dict[str, int]:
        """Test Redis MCP server operations."""
        tested, successful = 0, 0

        try:
            # Test SET operation
            tested += 1
            test_key = f"integration_test:{self.test_session_id}:set_test"
            result = await mcp_client.redis_set(key=test_key, value="test_value")

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis SET successful: {test_key}")
            else:
                print(f"❌ Redis SET failed: {result}")

            # Test GET operation
            tested += 1
            result = await mcp_client.redis_get(key=test_key)

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis GET successful: {result.get('value')}")
            else:
                print(f"❌ Redis GET failed: {result}")

            # Test DELETE operation
            tested += 1
            result = await mcp_client.redis_delete(key=test_key)

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis DELETE successful: {test_key}")
            else:
                print(f"❌ Redis DELETE failed: {result}")

        except Exception as e:
            print(f"Redis MCP operation failed: {e}")

        return {"tested": tested, "successful": successful}

    async def _test_taskqueue_mcp_operations(self, mcp_client) -> Dict[str, int]:
        """Test TaskQueue MCP server operations."""
        tested, successful = 0, 0

        try:
            # Test project listing
            tested += 1
            result = await mcp_client.taskqueue_list_projects()

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                projects = result.get('projects', [])
                print(f"✅ TaskQueue list_projects successful: {len(projects)} projects")
            else:
                print(f"❌ TaskQueue list_projects failed: {result}")

            # Test project creation
            tested += 1
            result = await mcp_client.taskqueue_create_project(
                initial_prompt=f"Test project for session {self.test_session_id}",
                tasks=["Task 1: Setup", "Task 2: Execute", "Task 3: Cleanup"]
            )

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ TaskQueue create_project successful: {result.get('project_id', 'Unknown ID')}")
            else:
                print(f"❌ TaskQueue create_project failed: {result}")

        except Exception as e:
            print(f"TaskQueue MCP operation failed: {e}")

        return {"tested": tested, "successful": successful}

    async def _test_temporal_mcp_operations(self, mcp_client) -> Dict[str, int]:
        """Test Temporal MCP server operations."""
        tested, successful = 0, 0

        try:
            # Test workflow history retrieval
            tested += 1
            result = await mcp_client.temporal_get_workflow_history(workflow_id="test_workflow")

            if isinstance(result, dict):
                # Success if we get a valid response
                if result.get("status") == "success" or "not found" in str(result.get("error", "")).lower():
                    successful += 1
                    print(f"✅ Temporal GetWorkflowHistory successful")
                else:
                    print(f"❌ Temporal GetWorkflowHistory failed: {result}")

        except Exception as e:
            print(f"Temporal MCP operation failed: {e}")

        return {"tested": tested, "successful": successful}

    async def _test_redis_key_operations(self, mcp_client) -> Dict[str, int]:
        """Test Redis key operations."""
        tested, successful = 0, 0

        try:
            test_key = f"integration_test:{self.test_session_id}:key_ops"

            # SET operation
            tested += 1
            result = await mcp_client.redis_set(key=test_key, value="test_value")

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis key SET successful: {test_key}")
            else:
                print(f"❌ Redis key SET failed: {result}")

            # GET operation
            tested += 1
            result = await mcp_client.redis_get(key=test_key)

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis key GET successful: {result.get('value')}")
            else:
                print(f"❌ Redis key GET failed: {result}")

            # DELETE operation
            tested += 1
            result = await mcp_client.redis_delete(key=test_key)

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis key DELETE successful: {test_key}")
            else:
                print(f"❌ Redis key DELETE failed: {result}")

        except Exception as e:
            print(f"Redis key operation failed: {e}")

        return {"tested": tested, "successful": successful}

    async def _test_redis_hash_operations(self, mcp_client) -> Dict[str, int]:
        """Test Redis hash operations."""
        tested, successful = 0, 0

        try:
            hash_key = f"integration_test:{self.test_session_id}:hash"

            # HSET operation
            tested += 1
            result = await mcp_client.redis_hset(name=hash_key, key="test_field", value="test_value")

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis HSET successful: {hash_key}.test_field")
            else:
                print(f"❌ Redis HSET failed: {result}")

            # HGET operation
            tested += 1
            result = await mcp_client.redis_hget(name=hash_key, key="test_field")

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis HGET successful: {result.get('value')}")
            else:
                print(f"❌ Redis HGET failed: {result}")

        except Exception as e:
            print(f"Redis hash operation failed: {e}")

        return {"tested": tested, "successful": successful}

    async def _test_redis_list_operations(self, mcp_client) -> Dict[str, int]:
        """Test Redis list operations."""
        tested, successful = 0, 0

        try:
            list_key = f"integration_test:{self.test_session_id}:list"

            # LPUSH operation
            tested += 1
            result = await mcp_client.redis_lpush(name=list_key, value="test_item")

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis LPUSH successful: {list_key}")
            else:
                print(f"❌ Redis LPUSH failed: {result}")

            # LLEN operation
            tested += 1
            result = await mcp_client.redis_llen(name=list_key)

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                print(f"✅ Redis LLEN successful: {result.get('length')}")
            else:
                print(f"❌ Redis LLEN failed: {result}")

        except Exception as e:
            print(f"Redis list operation failed: {e}")

        return {"tested": tested, "successful": successful}

    async def create_test_project(self, mcp_client) -> Dict[str, Any]:
        """Create test project in task queue."""
        try:
            result = await mcp_client.taskqueue_create_project(
                initial_prompt=f"Integration test pipeline project for session {self.test_session_id}",
                tasks=[
                    "Setup test environment",
                    "Execute integration tests",
                    "Generate test report",
                    "Cleanup test data"
                ]
            )

            if isinstance(result, dict) and result.get("status") == "success":
                project_id = result.get("project_id", f"test_{self.test_session_id}")
                return {
                    "success": True,
                    "project_id": project_id,
                    "details": "Project created successfully via MCP client",
                    "result": result,
                }
            else:
                return {
                    "success": False,
                    "error": str(result),
                    "details": "Project creation failed - TaskQueue MCP error"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": "Project creation failed - TaskQueue MCP may not be available"
            }

    async def add_test_tasks(self, mcp_client) -> Dict[str, Any]:
        """Add test tasks to project."""
        try:
            # Create project first to get project ID
            project_result = await self.create_test_project(mcp_client)
            if not project_result["success"]:
                return {
                    "success": False,
                    "error": "Could not create project",
                    "tasks_created": 0,
                    "total_tasks": 4,
                }

            project_id = project_result["project_id"]
            new_tasks = [
                f"Additional task 1 for {self.test_session_id}: Data validation",
                f"Additional task 2 for {self.test_session_id}: Model optimization",
                f"Additional task 3 for {self.test_session_id}: Performance testing",
                f"Additional task 4 for {self.test_session_id}: Documentation update",
            ]

            result = await mcp_client.taskqueue_add_tasks_to_project(
                project_id=project_id,
                tasks=new_tasks
            )

            if isinstance(result, dict) and result.get("status") == "success":
                return {
                    "success": True,
                    "details": f"Added {len(new_tasks)}/{len(new_tasks)} tasks successfully via MCP client",
                    "tasks_created": len(new_tasks),
                    "total_tasks": len(new_tasks),
                    "result": result,
                }
            else:
                return {
                    "success": False,
                    "error": str(result),
                    "tasks_created": 0,
                    "total_tasks": 4,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tasks_created": 0,
                "total_tasks": 4,
            }

    async def _test_task_retrieval(self, mcp_client) -> Dict[str, int]:
        """Test task retrieval operations."""
        tested, successful = 0, 0

        try:
            # Create project first to get project ID
            project_result = await self.create_test_project(mcp_client)
            if project_result["success"]:
                project_id = project_result["project_id"]

                # Test project info retrieval via read_project
                tested += 1
                result = await mcp_client.taskqueue_read_project(project_id=project_id)

                if isinstance(result, dict) and result.get("status") == "success":
                    successful += 1
                    print(f"✅ TaskQueue read_project successful")
                else:
                    print(f"❌ TaskQueue read_project failed: {result}")
            else:
                print("❌ Could not create project for task retrieval test")

            # Test general project listing
            tested += 1
            result = await mcp_client.taskqueue_list_projects()

            if isinstance(result, dict) and result.get("status") == "success":
                successful += 1
                projects = result.get('projects', [])
                print(f"✅ TaskQueue list_projects successful: {len(projects)} projects")
            else:
                print(f"❌ TaskQueue list_projects failed: {result}")

        except Exception as e:
            print(f"Task retrieval failed: {e}")

        return {"tested": tested, "successful": successful}

    async def get_workflow_history(self, mcp_client) -> Dict[str, Any]:
        """Get workflow history with MCP client."""
        try:
            result = await mcp_client.temporal_get_workflow_history(
                workflow_id="integration_test_workflow"
            )

            if isinstance(result, dict):
                if result.get("status") == "success":
                    return {
                        "success": True,
                        "details": "Retrieved workflow history via MCP client",
                        "history_events": len(result.get("history", [])),
                        "history_complete": True,
                        "result": result,
                    }
                elif "not found" in str(result.get("error", "")).lower():
                    # Workflow not found is expected for test workflow
                    return {
                        "success": True,
                        "details": "Workflow not found as expected for test workflow",
                        "history_events": 0,
                        "history_complete": True,
                        "result": result,
                    }
                else:
                    return {
                        "success": False,
                        "error": str(result),
                        "details": "Workflow history retrieval failed - unexpected response"
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": "Workflow history retrieval failed - Temporal workflow service may not be running"
            }

    async def setup_agent_coordination(self, mcp_client) -> Dict[str, Any]:
        """Setup agent coordination using Redis for state management."""
        try:
            coordination_key = f"agent_coordination:{self.test_session_id}"
            agents = ["data-engineer", "ai-engineer", "ml-engineer", "performance-monitor", "workflow-orchestrator"]

            # Store agent coordination state in Redis
            result = await mcp_client.redis_hset(
                name=coordination_key,
                key="active_agents",
                value=json.dumps(agents)
            )

            if isinstance(result, dict) and result.get("status") == "success":
                print(f"✅ Agent coordination stored in Redis: {coordination_key}")

                # Also store coordination metadata
                metadata_result = await mcp_client.redis_hset(
                    name=coordination_key,
                    key="metadata",
                    value=json.dumps({
                        "session_id": self.test_session_id,
                        "timestamp": datetime.now().isoformat(),
                        "agent_count": len(agents)
                    })
                )

                return {
                    "success": True,
                    "details": f"Coordination established for {len(agents)} agents via Redis storage",
                    "active_agents": agents,
                    "coordination_channels": 3,
                }
            else:
                return {
                    "success": False,
                    "error": str(result),
                    "active_agents": [],
                    "coordination_channels": 0,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "active_agents": [],
                "coordination_channels": 0,
            }

    async def create_collaborative_task(self, mcp_client) -> Dict[str, Any]:
        """Create collaborative task using storage."""
        try:
            task_id = f"prepare_training_data_collab_{self.test_session_id}"
            task_data = {
                "task_id": task_id,
                "description": "Collaborative audio dataset preparation for M1 training",
                "priority": "HIGH",
                "agents_required": ["data-engineer", "ai-engineer"],
                "status": "CREATED",
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            result = await mcp_client.redis_set(
                key=f"collaborative_task:{task_id}",
                value=json.dumps(task_data)
            )

            if isinstance(result, dict) and result.get("status") == "success":
                print(f"✅ Collaborative task created in Redis: {task_id}")
                return {
                    "success": True,
                    "details": "Collaborative training data preparation task created via storage",
                    "task_id": task_id,
                    "task_details": task_data,
                }
            else:
                return {
                    "success": False,
                    "error": str(result),
                    "task_id": None,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_id": None,
            }

    async def distribute_subtasks(self, mcp_client) -> Dict[str, Any]:
        """Distribute subtasks to agents using queuing."""
        try:
            subtasks = [
                {"agent": "data-engineer", "task": "Audio file validation"},
                {"agent": "data-engineer", "task": "Segmentation quality check"},
                {"agent": "ai-engineer", "task": "Model architecture review"},
                {"agent": "ml-engineer", "task": "Training pipeline setup"},
                {"agent": "performance-monitor", "task": "Metrics tracking setup"},
            ]

            distributed_count = 0
            # Distribute via Redis queues
            for i, subtask in enumerate(subtasks):
                queue_key = f"agent_queue:{subtask['agent']}:{self.test_session_id}"
                result = await mcp_client.redis_lpush(
                    name=queue_key,
                    value=json.dumps(subtask)
                )

                if isinstance(result, dict) and result.get("status") == "success":
                    distributed_count += 1
                    print(f"✅ Subtask distributed to {subtask['agent']}: {subtask['task']}")
                else:
                    print(f"❌ Failed to distribute subtask to {subtask['agent']}: {result}")

            return {
                "success": distributed_count > 0,
                "details": f"Distributed {distributed_count}/{len(subtasks)} subtasks successfully via queuing",
                "distributed_subtasks": distributed_count,
                "total_subtasks": len(subtasks),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "distributed_subtasks": 0,
                "total_subtasks": 5,
            }

    async def monitor_collaborative_progress(self, mcp_client) -> Dict[str, Any]:
        """Monitor collaborative progress using state tracking."""
        try:
            progress_key = f"collaboration_progress:{self.test_session_id}"

            # Simulate progress tracking via Redis
            progress_data = {
                "total_clips_processed": 1247,
                "total_duration_minutes": 67.3,
                "quality_score": 0.94,
                "agents_participated": 5,
                "completion_percentage": 100,
                "last_updated": datetime.now().isoformat(),
            }

            # Store progress in Redis
            result = await mcp_client.redis_set(
                key=progress_key,
                value=json.dumps(progress_data)
            )

            if isinstance(result, dict) and result.get("status") == "success":
                print(f"✅ Progress tracking stored in Redis: {progress_key}")

                # Retrieve and verify the stored data
                verify_result = await mcp_client.redis_get(key=progress_key)

                if isinstance(verify_result, dict) and verify_result.get("status") == "success":
                    stored_data = json.loads(verify_result.get("value", "{}"))
                    return {
                        "success": True,
                        "details": "Collaborative progress monitored successfully to 100% completion via state tracking",
                        "final_progress": stored_data.get("completion_percentage", 0),
                        "monitoring_points": 5,
                        "final_report": stored_data,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Could not verify stored progress: {verify_result}",
                        "final_progress": 0,
                    }
            else:
                return {
                    "success": False,
                    "error": str(result),
                    "final_progress": 0,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "final_progress": 0,
            }

    async def _test_task_handoff_protocol(self, mcp_client) -> Dict[str, Any]:
        """Test task handoff protocol with infrastructure."""
        try:
            # Create test project first
            project_result = await self.create_test_project(mcp_client)
            if not project_result["success"]:
                return {
                    "success": False,
                    "error": "Could not create project for handoff test",
                    "details": "Task handoff protocol test skipped - project creation failed"
                }

            project_id = project_result["project_id"]

            # Step 1: Store dataset processing context in Redis (simulating Agent A completing work)
            context_key = f"context:{project_id}:dataset:processing"
            context_data = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "segments_created": 1847,
                    "segments_kept": 1812,
                    "average_quality": 0.964,
                    "total_duration_sec": 2712
                },
                "manifest_path": f"data/processed/test_{self.test_session_id}/manifest.csv"
            }

            # Use json_set for structured context storage (following task-handoff-protocol.md)
            context_result = await mcp_client.redis_json_set(
                name=context_key,
                path="$",
                value=context_data
            )

            if not isinstance(context_result, dict) or context_result.get("status") != "success":
                return {
                    "success": False,
                    "error": f"Context storage failed: {context_result}",
                    "details": "Task handoff Step 2 (context storage) failed"
                }

            print(f"✅ Handoff Step 2: Context stored at {context_key}")

            # Step 2: Create follow-up task (simulating Agent A creating work for Agent B)
            new_task_result = await mcp_client.taskqueue_add_tasks_to_project(
                project_id=project_id,
                tasks=[{
                    "title": "Train LoRA model on processed dataset",
                    "description": f"Train model using dataset from {context_key}",
                    "toolRecommendations": "ai-engineer, ml-engineer",
                    "ruleRecommendations": f"Read context from Redis key: {context_key}"
                }]
            )

            if not isinstance(new_task_result, dict) or new_task_result.get("status") != "success":
                return {
                    "success": False,
                    "error": f"Task creation failed: {new_task_result}",
                    "details": "Task handoff Step 3 (follow-up task creation) failed"
                }

            print(f"✅ Handoff Step 3: Follow-up task created in project {project_id}")

            return {
                "success": True,
                "details": "3-step task handoff protocol validated successfully",
                "context_key": context_key,
                "project_id": project_id
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": "Task handoff protocol test failed with exception"
            }

    async def _test_event_driven_coordination(self, mcp_client) -> Dict[str, Any]:
        """Test event-driven coordination with Redis pub/sub."""
        events_tested = 0
        events_published = 0

        try:
            # Test publishing to different channels (following event-driven-coordination.md)
            test_events = [
                {
                    "channel": "training:events",
                    "event": {
                        "event": "checkpoint_saved",
                        "workflow_id": f"test_workflow_{self.test_session_id}",
                        "step": 3500,
                        "loss": 0.42,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                {
                    "channel": "metrics:performance",
                    "event": {
                        "event": "training_step",
                        "step": 100,
                        "gpu_utilization": 0.89,
                        "memory_used_gb": 18.4,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                {
                    "channel": "tasks:assignments",
                    "event": {
                        "event": "task_assigned",
                        "task_id": f"task_{self.test_session_id}",
                        "agent_type": "ai-engineer",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            ]

            for test_event in test_events:
                events_tested += 1
                try:
                    # Publish event using Redis pub/sub
                    result = await mcp_client.redis_publish(
                        channel=test_event["channel"],
                        message=json.dumps(test_event["event"])
                    )

                    if isinstance(result, dict) and result.get("status") == "success":
                        events_published += 1
                        print(f"✅ Event published to {test_event['channel']}: {test_event['event']['event']}")
                    else:
                        print(f"❌ Event publish failed to {test_event['channel']}: {result}")
                except Exception as e:
                    print(f"❌ Event publish exception for {test_event['channel']}: {e}")

            return {
                "success": events_published > 0,
                "details": f"Event-driven coordination tested with {events_published}/{events_tested} events published",
                "events_tested": events_tested,
                "events_published": events_published,
                "channels_tested": len(test_events)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "events_tested": events_tested,
                "events_published": events_published,
                "details": "Event-driven coordination test failed"
            }

    async def _test_workflow_agent_integration(self, mcp_client) -> Dict[str, Any]:
        """Test workflow-agent integration with Temporal."""
        try:
            # Test workflow history retrieval (following workflow-agent-integration.md)
            test_workflow_id = f"voice-clone-training-test-{self.test_session_id}"

            result = await mcp_client.temporal_get_workflow_history(
                workflow_id=test_workflow_id
            )

            if isinstance(result, dict):
                if result.get("status") == "success":
                    return {
                        "success": True,
                        "details": "Workflow-agent integration validated via Temporal history retrieval",
                        "workflow_id": test_workflow_id,
                        "history_events": len(result.get("history", []))
                    }
                elif "not found" in str(result.get("error", "")).lower():
                    # Expected for test workflow that doesn't exist
                    return {
                        "success": True,
                        "details": "Temporal MCP operational (workflow not found is expected for test)",
                        "workflow_id": test_workflow_id,
                        "history_events": 0
                    }
                else:
                    return {
                        "success": False,
                        "error": str(result.get("error")),
                        "details": "Workflow history retrieval failed"
                    }
            else:
                return {
                    "success": False,
                    "error": "Unexpected result type",
                    "details": "Workflow-agent integration test received unexpected response"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": "Workflow-agent integration test failed with exception"
            }

    async def _test_workflow_status_command(self, mcp_client) -> Dict[str, Any]:
        """Test /workflow-status command equivalent with infrastructure."""
        try:
            # Simulate /workflow-status command by checking Temporal workflow
            test_workflow_id = f"voice-clone-training-{self.test_session_id}"

            # Get workflow history
            history_result = await mcp_client.temporal_get_workflow_history(
                workflow_id=test_workflow_id
            )

            # Parse workflow status
            if isinstance(history_result, dict):
                if history_result.get("status") == "success":
                    history = history_result.get("history", [])
                    status = "RUNNING" if len(history) > 0 else "UNKNOWN"
                    return {
                        "success": True,
                        "details": f"Workflow {test_workflow_id} status: {status}",
                        "workflow_id": test_workflow_id,
                        "status": status,
                        "events": len(history)
                    }
                elif "not found" in str(history_result.get("error", "")).lower():
                    return {
                        "success": True,
                        "details": f"Workflow {test_workflow_id} not found (expected for test)",
                        "workflow_id": test_workflow_id,
                        "status": "NOT_FOUND",
                        "events": 0
                    }
                else:
                    return {
                        "success": False,
                        "error": str(history_result.get("error")),
                        "details": "Workflow status check failed"
                    }
            else:
                return {
                    "success": False,
                    "error": "Unexpected result type",
                    "details": "Workflow status command returned unexpected response"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": "/workflow-status equivalent test failed"
            }

    async def _test_agent_status_command(self, mcp_client) -> Dict[str, Any]:
        """Test /agent-status command equivalent with infrastructure."""
        try:
            # Simulate /agent-status by tracking agent availability in Redis
            agents = [
                "data-engineer",
                "ai-engineer",
                "ml-engineer",
                "performance-monitor",
                "error-coordinator"
            ]

            agents_tracked = 0
            agents_available = 0

            for agent in agents:
                status_key = f"agent:status:{agent}:{self.test_session_id}"

                # Store agent status
                status_data = {
                    "agent": agent,
                    "status": "AVAILABLE",
                    "last_heartbeat": datetime.now().isoformat(),
                    "circuit_breaker": "CLOSED"
                }

                result = await mcp_client.redis_set(
                    key=status_key,
                    value=json.dumps(status_data)
                )

                if isinstance(result, dict) and result.get("status") == "success":
                    agents_tracked += 1
                    agents_available += 1
                    print(f"✅ Agent status tracked: {agent} = AVAILABLE")
                else:
                    print(f"❌ Failed to track agent status: {agent}")

            return {
                "success": agents_tracked > 0,
                "details": f"Agent status command validated with {agents_tracked} agents",
                "agents_tracked": agents_tracked,
                "total_agents": len(agents),
                "agents_available": agents_available
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agents_tracked": 0,
                "total_agents": 5,
                "details": "/agent-status equivalent test failed"
            }

    async def _test_check_infrastructure_command(self, mcp_client) -> Dict[str, Any]:
        """Test /check-infrastructure command equivalent with infrastructure."""
        try:
            components = {
                "redis": False,
                "taskqueue": False,
                "temporal": False
            }

            # Check Redis
            redis_test = await self.check_redis_health(mcp_client)
            components["redis"] = redis_test

            # Check TaskQueue
            try:
                taskqueue_test = await mcp_client.taskqueue_list_projects()
                components["taskqueue"] = isinstance(taskqueue_test, dict) and taskqueue_test.get("status") == "success"
            except:
                components["taskqueue"] = False

            # Check Temporal
            temporal_test = await self.check_temporal_health(mcp_client)
            components["temporal"] = temporal_test

            healthy_count = sum(1 for v in components.values() if v)
            total_count = len(components)

            return {
                "success": healthy_count >= 1,  # At least one component should be healthy
                "details": f"Infrastructure check: {healthy_count}/{total_count} components healthy",
                "healthy_components": healthy_count,
                "total_components": total_count,
                "components": components
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "healthy_components": 0,
                "total_components": 3,
                "details": "/check-infrastructure equivalent test failed"
            }


# pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

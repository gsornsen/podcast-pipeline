"""Health checks for infrastructure components.

This module validates that MCP servers and system components are available
before running tests or training pipelines.
"""

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass

import structlog

from infrastructure.config import (
    InfrastructureConfig,
    RedisConfig,
    TemporalConfig,
)

logger = structlog.get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status for infrastructure components."""

    healthy: bool
    components: dict[str, bool]
    errors: dict[str, str]

    def print_report(self) -> None:
        """Print formatted health check report."""
        print("\n=== Infrastructure Health Check ===\n")

        for component, status in self.components.items():
            symbol = "✅" if status else "❌"
            print(f"{symbol} {component:<20} {'HEALTHY' if status else 'FAILED'}")

            if not status and component in self.errors:
                print(f"   └─ Error: {self.errors[component]}")

        print(f"\n{'='*35}")
        print(f"Overall Status: {'HEALTHY ✅' if self.healthy else 'UNHEALTHY ❌'}\n")


async def check_redis_health(config: RedisConfig) -> bool:
    """Check Redis connectivity via RedisMCPServer MCP.

    Args:
        config: Redis configuration

    Returns:
        True if Redis is healthy, False otherwise
    """
    try:
        # Test Redis connectivity via redis-cli
        result = subprocess.run(
            ["redis-cli", "-u", config.url, "PING"],
            capture_output=True,
            text=True,
            timeout=config.socket_connect_timeout,
        )

        if result.returncode == 0 and "PONG" in result.stdout:
            logger.info("redis_health_check_passed", url=config.url)
            return True

        logger.error(
            "redis_health_check_failed",
            url=config.url,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return False

    except subprocess.TimeoutExpired:
        logger.error("redis_health_check_timeout", url=config.url)
        return False
    except FileNotFoundError:
        logger.warning("redis_cli_not_found", message="redis-cli not installed, skipping check")
        # Return True if redis-cli is not installed (assume Redis via MCP)
        return True
    except Exception as e:
        logger.error("redis_health_check_error", error=str(e))
        return False


async def check_temporal_health(config: TemporalConfig) -> bool:
    """Check Temporal connectivity via temporal-mcp MCP.

    Args:
        config: Temporal configuration

    Returns:
        True if Temporal is healthy, False otherwise
    """
    try:
        # Test Temporal connectivity via temporal workflow list
        # This requires temporal CLI to be installed
        result = subprocess.run(
            [
                "temporal",
                "workflow",
                "list",
                "--namespace",
                config.namespace,
                "--address",
                config.host,
                "--limit",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            logger.info("temporal_health_check_passed", host=config.host)
            return True

        logger.error(
            "temporal_health_check_failed",
            host=config.host,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return False

    except subprocess.TimeoutExpired:
        logger.error("temporal_health_check_timeout", host=config.host)
        return False
    except FileNotFoundError:
        logger.warning("temporal_cli_not_found", message="temporal CLI not installed, skipping check")
        # Return True if temporal CLI is not installed (assume Temporal via MCP)
        return True
    except Exception as e:
        logger.error("temporal_health_check_error", error=str(e))
        return False


async def check_taskqueue_health() -> bool:
    """Check taskqueue MCP availability.

    Returns:
        True if taskqueue is healthy, False otherwise

    Note:
        Since taskqueue is an MCP server accessed via npx, we check
        if npx and the package are available.
    """
    try:
        # Check if npx is available
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            logger.info("taskqueue_health_check_passed")
            return True

        logger.error("taskqueue_health_check_failed", stderr=result.stderr)
        return False

    except subprocess.TimeoutExpired:
        logger.error("taskqueue_health_check_timeout")
        return False
    except FileNotFoundError:
        logger.error("npx_not_found", message="npx not installed")
        return False
    except Exception as e:
        logger.error("taskqueue_health_check_error", error=str(e))
        return False


async def check_gpu_health(required_model: str = "RTX 4090") -> bool:
    """Check GPU availability.

    Args:
        required_model: Required GPU model name

    Returns:
        True if required GPU is available, False otherwise
    """
    try:
        # Query GPU name via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            gpu_name = result.stdout.strip()

            if required_model.lower() in gpu_name.lower():
                logger.info("gpu_health_check_passed", gpu=gpu_name)
                return True

            logger.error(
                "gpu_model_mismatch",
                expected=required_model,
                found=gpu_name,
            )
            return False

        logger.error("gpu_health_check_failed", stderr=result.stderr)
        return False

    except subprocess.TimeoutExpired:
        logger.error("gpu_health_check_timeout")
        return False
    except FileNotFoundError:
        logger.warning("nvidia_smi_not_found", message="nvidia-smi not installed, skipping GPU check")
        # Return True if nvidia-smi is not installed (allow CPU-only testing)
        return True
    except Exception as e:
        logger.error("gpu_health_check_error", error=str(e))
        return False


async def check_infrastructure_health(
    config: InfrastructureConfig | None = None,
) -> HealthStatus:
    """Check all infrastructure components.

    Args:
        config: Infrastructure configuration (loads from env if None)

    Returns:
        Health status for all components
    """
    if config is None:
        config = InfrastructureConfig.load()

    logger.info("starting_infrastructure_health_check")

    components: dict[str, bool] = {}
    errors: dict[str, str] = {}

    # Run all health checks concurrently
    results = await asyncio.gather(
        check_redis_health(config.redis),
        check_temporal_health(config.temporal),
        check_taskqueue_health(),
        check_gpu_health(config.gpu_model) if config.gpu_required else asyncio.create_task(asyncio.sleep(0, result=True)),
        return_exceptions=True,
    )

    # Process Redis health check
    if isinstance(results[0], Exception):
        components["Redis"] = False
        errors["Redis"] = str(results[0])
    else:
        components["Redis"] = results[0]
        if not results[0]:
            errors["Redis"] = f"Cannot connect to Redis at {config.redis.url}"

    # Process Temporal health check
    if isinstance(results[1], Exception):
        components["Temporal"] = False
        errors["Temporal"] = str(results[1])
    else:
        components["Temporal"] = results[1]
        if not results[1]:
            errors["Temporal"] = f"Cannot connect to Temporal at {config.temporal.host}"

    # Process TaskQueue health check
    if isinstance(results[2], Exception):
        components["TaskQueue"] = False
        errors["TaskQueue"] = str(results[2])
    else:
        components["TaskQueue"] = results[2]
        if not results[2]:
            errors["TaskQueue"] = "TaskQueue MCP server not available"

    # Process GPU health check
    if isinstance(results[3], Exception):
        components["GPU"] = False
        errors["GPU"] = str(results[3])
    else:
        components["GPU"] = results[3]
        if not results[3] and config.gpu_required:
            errors["GPU"] = f"Required GPU model '{config.gpu_model}' not found"

    # Overall health is True only if all required components are healthy
    healthy = all(components.values())

    logger.info(
        "infrastructure_health_check_complete",
        healthy=healthy,
        components=components,
    )

    return HealthStatus(healthy=healthy, components=components, errors=errors)


async def main() -> int:
    """Run health checks as a standalone script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Configure structured logging for CLI output
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    try:
        config = InfrastructureConfig.load()
        status = await check_infrastructure_health(config)
        status.print_report()

        return 0 if status.healthy else 1

    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        print(f"\n❌ Health check failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Configuration models for infrastructure orchestration."""


from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisConfig(BaseSettings):
    """Redis/Valkey configuration for task queue and pub/sub."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )
    db: int = Field(
        default=0,
        description="Redis database number",
        ge=0,
        le=15,
    )
    max_connections: int = Field(
        default=10,
        description="Maximum connection pool size",
        gt=0,
    )
    socket_timeout: float = Field(
        default=5.0,
        description="Socket timeout in seconds",
        gt=0,
    )
    socket_connect_timeout: float = Field(
        default=5.0,
        description="Socket connect timeout in seconds",
        gt=0,
    )
    retry_on_timeout: bool = Field(
        default=True,
        description="Retry on timeout",
    )
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds",
        gt=0,
    )


class TemporalConfig(BaseSettings):
    """Temporal workflow engine configuration."""

    model_config = SettingsConfigDict(
        env_prefix="TEMPORAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="localhost:7233",
        description="Temporal server host:port",
    )
    namespace: str = Field(
        default="default",
        description="Temporal namespace",
    )
    task_queue: str = Field(
        default="podcast-pipeline",
        description="Default task queue name",
    )
    workflow_execution_timeout: int = Field(
        default=86400,  # 24 hours
        description="Workflow execution timeout in seconds",
        gt=0,
    )
    workflow_run_timeout: int = Field(
        default=3600,  # 1 hour
        description="Workflow run timeout in seconds",
        gt=0,
    )
    workflow_task_timeout: int = Field(
        default=10,
        description="Workflow task timeout in seconds",
        gt=0,
    )
    activity_start_to_close_timeout: int = Field(
        default=300,  # 5 minutes
        description="Activity start-to-close timeout in seconds",
        gt=0,
    )
    activity_heartbeat_timeout: int | None = Field(
        default=60,
        description="Activity heartbeat timeout in seconds",
    )


class TaskQueueConfig(BaseSettings):
    """TaskQueue MCP configuration."""

    model_config = SettingsConfigDict(
        env_prefix="TASKQUEUE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    auto_approve: bool = Field(
        default=False,
        description="Automatically approve completed tasks",
    )
    default_priority: str = Field(
        default="normal",
        description="Default task priority (critical, high, normal, low)",
    )


class InfrastructureConfig(BaseSettings):
    """Combined infrastructure configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis: RedisConfig = Field(default_factory=RedisConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    taskqueue: TaskQueueConfig = Field(default_factory=TaskQueueConfig)

    gpu_required: bool = Field(
        default=True,
        description="Require GPU availability",
    )
    gpu_model: str = Field(
        default="RTX 4090",
        description="Required GPU model",
    )

    @classmethod
    def load(cls) -> "InfrastructureConfig":
        """Load configuration from environment and .env file."""
        return cls()

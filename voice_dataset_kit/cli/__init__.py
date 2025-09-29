"""Command-line interface modules for voice_dataset_kit.

Available commands:
- build_dataset: Convert audio files into segmented datasets
- reviewer: Review and manage processed datasets

Usage:
    uv run python -m voice_dataset_kit.cli.build_dataset --help
    uv run python -m voice_dataset_kit.cli.reviewer --help
"""

# Note: Modules are imported on-demand to avoid sys.modules warnings
__all__ = ["build_dataset", "reviewer"]
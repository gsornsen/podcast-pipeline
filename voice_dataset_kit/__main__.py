#!/usr/bin/env python3
"""Main entry point for voice_dataset_kit module.

This allows running the package as a module with:
    uv run python -m voice_dataset_kit

Provides helpful guidance for CLI usage.
"""

import sys
from pathlib import Path


def main() -> None:
    """Main entry point showing available commands."""
    print("Voice Dataset Kit - Convert long recordings into clean, segmented utterances")
    print()
    print("Available commands:")
    print("  uv run python -m voice_dataset_kit.cli.build_dataset  # Build dataset from audio files")
    print("  uv run python -m voice_dataset_kit.cli.reviewer       # Review processed datasets")
    print()
    print("Or after installation:")
    print("  voice-dataset-build   # Build dataset from audio files")
    print("  voice-dataset-review  # Review processed datasets")
    print()
    print("For help with specific commands, add --help:")
    print("  uv run python -m voice_dataset_kit.cli.build_dataset --help")
    print()
    print("Note: The module name uses underscores (voice_dataset_kit), not hyphens.")


if __name__ == "__main__":
    main()
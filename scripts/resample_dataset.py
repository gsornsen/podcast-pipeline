#!/usr/bin/env python3
"""
Resample audio dataset from 16kHz to 24kHz.

This script resamples all audio files in a voice-dataset-kit output directory
from 16kHz to 24kHz using high-quality resampling (librosa kaiser_best).
Preserves train/val/test splits and updates metadata.jsonl files.

Usage:
    uv run python scripts/resample_dataset.py \
        --input ~/Downloads/gerald_output_audio/ \
        --output ~/Downloads/gerald_output_audio_24k/
"""

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
from tqdm import tqdm


def resample_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = 24000,
    quality: str = "kaiser_best",
) -> dict[str, Any]:
    """
    Resample a single audio file to target sample rate.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        target_sr: Target sample rate in Hz
        quality: Resampling quality ('kaiser_best', 'kaiser_fast', etc.)

    Returns:
        Dict with resampling stats (duration, sample_rate, etc.)
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=target_sr, res_type=quality
        )

    # Save resampled audio (16-bit PCM, mono)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, target_sr, subtype="PCM_16")

    return {
        "duration_sec": len(audio) / target_sr,
        "sample_rate": target_sr,
        "original_sr": sr,
        "num_samples": len(audio),
    }


def update_metadata_entry(
    entry: dict[str, Any], target_sr: int = 24000
) -> dict[str, Any]:
    """
    Update a metadata.jsonl entry with new sample rate.

    Args:
        entry: Original metadata entry
        target_sr: Target sample rate

    Returns:
        Updated metadata entry
    """
    updated = entry.copy()
    updated["sample_rate"] = target_sr
    return updated


def resample_split(
    input_dir: Path,
    output_dir: Path,
    split_name: str,
    target_sr: int = 24000,
    quality: str = "kaiser_best",
) -> int:
    """
    Resample all audio files in a split (train/val/test).

    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        split_name: Name of split ('train', 'val', 'test')
        target_sr: Target sample rate
        quality: Resampling quality

    Returns:
        Number of files processed
    """
    input_split = input_dir / split_name
    output_split = output_dir / split_name

    if not input_split.exists():
        print(f"⚠️  Split {split_name}/ not found, skipping")
        return 0

    # Load metadata
    metadata_path = input_split / "metadata.jsonl"
    if not metadata_path.exists():
        print(f"⚠️  {split_name}/metadata.jsonl not found, skipping")
        return 0

    # Read all metadata entries
    entries = []
    with open(metadata_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    print(f"\n📁 Processing {split_name} split ({len(entries)} files)...")

    # Process each file
    updated_entries = []
    for entry in tqdm(entries, desc=f"Resampling {split_name}"):
        # Get audio file path (relative to split directory)
        audio_file = entry["audio"]

        # Handle both relative paths (train/file.wav) and simple names (file.wav)
        if audio_file.startswith(f"{split_name}/"):
            audio_filename = audio_file.split("/", 1)[1]
        else:
            audio_filename = audio_file

        input_audio = input_split / audio_filename
        output_audio = output_split / audio_filename

        if not input_audio.exists():
            print(f"⚠️  Warning: {input_audio} not found, skipping")
            continue

        # Resample audio
        try:
            resample_audio(input_audio, output_audio, target_sr, quality)

            # Update metadata entry
            updated_entry = update_metadata_entry(entry, target_sr)

            # Update audio path to be relative
            updated_entry["audio"] = f"{split_name}/{audio_filename}"

            updated_entries.append(updated_entry)
        except Exception as e:
            print(f"❌ Error processing {audio_filename}: {e}")
            continue

    # Write updated metadata
    output_metadata = output_split / "metadata.jsonl"
    with open(output_metadata, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ {split_name}: {len(updated_entries)} files resampled")

    return len(updated_entries)


def validate_resampling(output_dir: Path, target_sr: int = 24000) -> dict[str, Any]:
    """
    Validate resampled dataset.

    Args:
        output_dir: Output dataset directory
        target_sr: Expected sample rate

    Returns:
        Validation report
    """
    print("\n🔍 Validating resampled dataset...")

    validation = {"splits": {}, "total_files": 0, "all_valid": True, "errors": []}

    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if not split_dir.exists():
            continue

        metadata_path = split_dir / "metadata.jsonl"
        if not metadata_path.exists():
            continue

        # Check metadata
        with open(metadata_path) as f:
            entries = [json.loads(line) for line in f]

        split_valid = True
        sample_rates = set()

        # Validate sample files (check first 3)
        for _, entry in enumerate(entries[:3]):
            audio_path = split_dir / entry["audio"].split("/", 1)[1]

            try:
                # Load just metadata (fast)
                info = sf.info(str(audio_path))
                sample_rates.add(info.samplerate)

                if info.samplerate != target_sr:
                    validation["errors"].append(
                        f"{split_name}/{entry['audio']}: Expected {target_sr}Hz, got {info.samplerate}Hz"
                    )
                    split_valid = False
                    validation["all_valid"] = False

                # Verify 16-bit PCM mono
                if info.subtype != "PCM_16":
                    validation["errors"].append(
                        f"{split_name}/{entry['audio']}: Expected PCM_16, got {info.subtype}"
                    )
                    split_valid = False

                if info.channels != 1:
                    validation["errors"].append(
                        f"{split_name}/{entry['audio']}: Expected mono, got {info.channels} channels"
                    )
                    split_valid = False

            except Exception as e:
                validation["errors"].append(f"{split_name}/{entry['audio']}: {e}")
                split_valid = False
                validation["all_valid"] = False

        validation["splits"][split_name] = {
            "count": len(entries),
            "valid": split_valid,
            "sample_rates": list(sample_rates),
        }
        validation["total_files"] += len(entries)

    return validation


def copy_supplementary_files(input_dir: Path, output_dir: Path):
    """Copy run_summary.json and any other non-audio files."""
    print("\n📋 Copying supplementary files...")

    # Copy run_summary.json if exists
    if (input_dir / "run_summary.json").exists():
        # Load, update sample rate, save
        with open(input_dir / "run_summary.json") as f:
            summary = json.load(f)

        # Update config sample rate if present
        if "config" in summary:
            summary["config"]["sr"] = 24000

        # Add resampling note
        summary["resampled_from_16khz"] = True
        summary["resampling_quality"] = "kaiser_best"

        with open(output_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("✅ Updated run_summary.json with new sample rate")


def main():
    parser = argparse.ArgumentParser(
        description="Resample audio dataset from 16kHz to 24kHz"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset directory (e.g., ~/Downloads/gerald_output_audio/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory (e.g., ~/Downloads/gerald_output_audio_24k/)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=24000,
        help="Target sample rate in Hz (default: 24000)",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="kaiser_best",
        choices=["kaiser_best", "kaiser_fast", "scipy", "polyphase"],
        help="Resampling quality (default: kaiser_best)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing resampled dataset",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    print("🎵 Audio Dataset Resampling Tool")
    print("═" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target: {args.target_sr} Hz")
    print(f"Quality: {args.quality}")
    print("═" * 60)

    # Validation-only mode
    if args.validate_only:
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            return 1

        validation = validate_resampling(output_dir, args.target_sr)

        print("\n📊 Validation Results:")
        print(f"Total files: {validation['total_files']}")
        for split, stats in validation["splits"].items():
            status = "✅" if stats["valid"] else "❌"
            print(
                f"  {status} {split}: {stats['count']} files, SR: {stats['sample_rates']}"
            )

        if validation["errors"]:
            print("\n❌ Errors found:")
            for error in validation["errors"]:
                print(f"  • {error}")
            return 1

        print("\n✅ All validations passed!")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resample each split
    total_processed = 0
    for split_name in ["train", "val", "test"]:
        count = resample_split(
            input_dir, output_dir, split_name, args.target_sr, args.quality
        )
        total_processed += count

    # Copy supplementary files
    copy_supplementary_files(input_dir, output_dir)

    # Validate output
    validation = validate_resampling(output_dir, args.target_sr)

    # Print summary
    print("\n" + "═" * 60)
    print("📊 Resampling Summary")
    print("═" * 60)
    print(f"Total files processed: {total_processed}")
    print(f"Total files in output: {validation['total_files']}")

    for split, stats in validation["splits"].items():
        status = "✅" if stats["valid"] else "❌"
        print(f"  {status} {split}: {stats['count']} files")

    if validation["errors"]:
        print("\n⚠️  Validation warnings:")
        for error in validation["errors"][:5]:  # Show first 5
            print(f"  • {error}")
        if len(validation["errors"]) > 5:
            print(f"  ... and {len(validation['errors']) - 5} more")
        return 1

    print("\n✅ Dataset resampling complete!")
    print(f"Output: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())

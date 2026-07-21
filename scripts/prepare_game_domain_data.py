#!/usr/bin/env python3
"""
Prepare game data for LayerCake portable game domain.

Takes game text files and converts them to JSONL format suitable for:
1. Training a portable domain decoder on game-specific language
2. Integrating with game FAQ learning system
3. Preserving domain state for player interactions

Game FAQ will learn from:
- Initial domain training (your game text files)
- Continuous feedback during gameplay (player interactions)
- Retraining cycles to adapt to player patterns
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
from pathlib import Path


def load_game_texts(
    game_dir: Path,
    *,
    exclude: set[Path] | None = None,
) -> Iterator[str]:
    """Load all game text files from directory."""
    if not game_dir.exists():
        raise FileNotFoundError(f"Game directory not found: {game_dir}")

    print(f"Scanning game directory: {game_dir}")

    # Support common game text formats
    extensions = {".txt", ".md", ".json", ".csv", ".jsonl"}
    excluded = {path.resolve() for path in (exclude or set())}

    for file_path in sorted(game_dir.rglob("*")):
        if file_path.resolve() in excluded:
            continue
        if file_path.suffix.lower() in extensions and file_path.is_file():
            try:
                if file_path.suffix.lower() == ".json":
                    with file_path.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                        if isinstance(data, dict):
                            yield json.dumps(data)
                        elif isinstance(data, list):
                            for item in data:
                                yield json.dumps(item)
                elif file_path.suffix.lower() == ".jsonl":
                    with file_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if line:
                                yield line
                else:
                    # TXT, MD, CSV
                    with file_path.open(
                        "r", encoding="utf-8", errors="replace"
                    ) as handle:
                        content = handle.read()
                        if content.strip():
                            yield content
            except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue


def prepare_game_domain_data(
    game_dir: Path,
    output_jsonl: Path,
    max_docs: int | None = None,
) -> int:
    """
    Convert game texts to JSONL format for portable domain training.

    Each document becomes a game context that the domain learns to understand:
    - Dialogue/narrative from game scripts
    - Quest descriptions and objectives
    - Character bios and interactions
    - Item/location descriptions
    - Player-facing FAQ content
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    doc_count = 0

    with output_jsonl.open("w", encoding="utf-8") as outf:
        for i, text in enumerate(
            load_game_texts(game_dir, exclude={output_jsonl})
        ):
            if max_docs and i >= max_docs:
                break

            # Skip empty documents
            if not text or not text.strip():
                continue

            # Create JSONL document
            doc = {
                "text": text.strip(),
                "source": "game_domain",
                "doc_id": i,
            }

            # Try to infer document type from content
            text_lower = text.lower()
            if any(x in text_lower for x in ['quest', 'objective', 'mission']):
                doc["type"] = "quest"
            elif any(x in text_lower for x in ['dialogue', 'said', 'says', 'asked']):
                doc["type"] = "dialogue"
            elif any(x in text_lower for x in ['item', 'weapon', 'armor', 'equipment']):
                doc["type"] = "item"
            elif any(x in text_lower for x in ['location', 'area', 'zone', 'region']):
                doc["type"] = "location"
            elif any(x in text_lower for x in ['faq', 'help', 'guide', 'howto']):
                doc["type"] = "faq"
            else:
                doc["type"] = "narrative"

            outf.write(json.dumps(doc) + "\n")
            doc_count += 1

            if doc_count % 100 == 0:
                print(f"Prepared {doc_count} documents...")

    print(f"\nCompleted: {doc_count} documents written to {output_jsonl}")
    return doc_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare game data for LayerCake portable game domain"
    )
    parser.add_argument(
        "--game-dir",
        required=True,
        help="Directory containing game text files (TXT, MD, JSON, CSV, JSONL)"
    )
    parser.add_argument(
        "--output",
        default="data/game_domain_training.jsonl",
        help="Output JSONL file for training"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to process (default: all)"
    )
    args = parser.parse_args()

    game_dir = Path(args.game_dir)
    output_jsonl = Path(args.output)

    print("=" * 60)
    print("Game Domain Data Preparation")
    print("=" * 60)
    print(f"Game directory: {game_dir}")
    print(f"Output file:    {output_jsonl}")

    doc_count = prepare_game_domain_data(game_dir, output_jsonl, args.max_docs)

    print(f"\n{'='*60}")
    print("Next steps:")
    print("  1. Train portable domain on: python scripts/train_portable_domain_decoder.py")
    print(f"     --decoder-data {output_jsonl}")
    print(f"     --source-core runs_experiment/layercake_250m_english_core.pt")
    print(f"     --output runs_experiment/portable_game_domain.pt")
    print("\n  2. Integrate with core:")
    print("     LayerCakeRuntime will install both core and game domain")
    print("     Game domain augments English understanding with game-specific context")
    print("\n  3. FAQ Learning:")
    print("     Player questions → game domain → FAQ answers")
    print("     Player feedback → retrain game domain → continuous improvement")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

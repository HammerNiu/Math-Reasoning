"""
Train the Process Preference Model (PPM) on collected trajectory data.

Usage:
    # First collect trajectories:
    python tools/collect_trajectories.py --model deepseek --output data/trajectories.jsonl

    # Then train the PPM:
    python tools/train_ppm.py --data data/trajectories.jsonl --output checkpoints/ppm.pt

    # With validation split and custom epochs:
    python tools/train_ppm.py --data data/trajectories.jsonl --epochs 100 --val-split 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ppm import PPMConfig, PPMTrainer, ProcessPreferenceModel
from src.model.model_interface import ModelFactory


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_preference_pairs(data_path: Path) -> List[Dict[str, str]]:
    """Flatten all preference_pairs from a trajectories JSONL file."""
    pairs: List[Dict[str, str]] = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record: Dict[str, Any] = json.loads(line)
            for pair in record.get("preference_pairs", []):
                preferred = (pair.get("preferred") or "").strip()
                non_preferred = (pair.get("non_preferred") or "").strip()
                if preferred and non_preferred:
                    pairs.append({"preferred": preferred, "non_preferred": non_preferred})
    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}\n"
                 "Run tools/collect_trajectories.py first to generate training data.")

    all_pairs = load_preference_pairs(data_path)
    if not all_pairs:
        sys.exit("No preference pairs found in the data file. "
                 "Make sure collect_trajectories.py produced non-empty output.")

    print(f"Loaded {len(all_pairs)} preference pairs from {data_path}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * args.val_split)) if len(all_pairs) > 4 else 0
    val_data = all_pairs[:n_val]
    train_data = all_pairs[n_val:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    # Embedder — use free local sentence-transformers (all-MiniLM-L6-v2, dim=384)
    # No API key required.
    from src.model.model_interface import LocalEmbedder
    embedder = LocalEmbedder.get()
    print(f"Using local embedder: all-MiniLM-L6-v2  (dim={embedder.dim})")

    # Build PPM — input_dim must match embedding dimension
    ppm_config = PPMConfig(
        input_dim=embedder.dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    ppm = ProcessPreferenceModel(ppm_config)
    trainer = PPMTrainer(ppm)

    print(f"Training for {args.epochs} epochs …")
    history = trainer.train(
        train_data,
        embedder,
        num_epochs=args.epochs,
        validation_data=val_data if val_data else None,
    )

    # Save checkpoint
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ppm.save_model(str(output))

    # Report
    train_losses = history["train_losses"]
    val_losses = history.get("val_losses", [])
    print(f"\nTraining complete.")
    print(f"  Final train loss : {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Final val loss   : {val_losses[-1]:.4f}")
    print(f"  Checkpoint saved : {output}")
    print(f"\nTo use the PPM in MCTS, pass --ppm-checkpoint {output} to your script,")
    print(f"or load it with: mcts.set_ppm(ppm)  after ppm.load_model('{output}')")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to trajectories JSONL (from collect_trajectories.py)")
    parser.add_argument("--output", default="checkpoints/ppm.pt", help="Where to save the trained PPM checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data to use for validation")
    parser.add_argument("--hidden-dim", type=int, default=256, help="PPM hidden layer dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

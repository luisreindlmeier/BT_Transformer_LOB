"""
Central training orchestration script.

Allows:
  - Running single or multiple models
  - Training on single or multiple tickers
  - Controlling dataset fraction (default 10%)
  - Central configuration management

Usage:
  # Train LiT on CSCO with 10% data
  python run_training.py --model lit --ticker CSCO --data-fraction 0.1

  # Train all models on CSCO with 5% data
  python run_training.py --model all --ticker CSCO --data-fraction 0.05

  # Train baseline and deeplob on CSCO with 100% data
  python run_training.py --model baseline deeplob --ticker CSCO --data-fraction 1.0
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Supported models
MODELS = {
    "baseline": "train_baseline.py",
    "deeplob": "train_deeplob.py",
    "lit": "train_lit.py",
    "tlob": "train_tlob.py",
}

# Supported tickers (extend as needed)
TICKERS = ["CSCO", "AAPL", "GOOG", "INTC"]

TRAINING_DIR = Path(__file__).resolve().parent


def validate_args(args) -> tuple[List[str], List[str]]:
    """Validate and normalize arguments."""
    # Validate models
    if args.model == ["all"]:
        models = list(MODELS.keys())
    else:
        models = args.model
        for m in models:
            if m not in MODELS:
                print(f"ERROR: Unknown model '{m}'. Available: {list(MODELS.keys())}")
                sys.exit(1)

    # Validate tickers
    if args.ticker == ["all"]:
        tickers = TICKERS
    else:
        tickers = args.ticker
        for t in tickers:
            if t not in TICKERS:
                print(f"WARNING: Ticker '{t}' not in predefined list {TICKERS}")
                print(f"Continuing anyway (data directory must exist)")

    # Validate data fraction
    if not (0.0 < args.data_fraction <= 1.0):
        print(f"ERROR: data_fraction must be in (0, 1], got {args.data_fraction}")
        sys.exit(1)

    return models, tickers


def update_trainer_config(trainer_path: Path, ticker: str, data_fraction: float):
    """Update TICKER and DATA_FRACTION in trainer script."""
    with open(trainer_path, "r") as f:
        content = f.read()

    # Replace TICKER
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("TICKER = "):
            lines[i] = f'TICKER = "{ticker}"'
        elif line.startswith("DATA_FRACTION = "):
            lines[i] = f"DATA_FRACTION = {data_fraction}"

    updated_content = "\n".join(lines)

    with open(trainer_path, "w") as f:
        f.write(updated_content)

    print(f"  Updated: TICKER={ticker}, DATA_FRACTION={data_fraction}")


def run_trainer(model: str, ticker: str, data_fraction: float) -> int:
    """Run a single trainer with given config."""
    trainer_script = TRAINING_DIR / MODELS[model]

    if not trainer_script.exists():
        print(f"ERROR: {trainer_script} not found")
        return 1

    print(f"\n{'='*80}")
    print(f"Training: {model.upper()} on {ticker} ({data_fraction*100:.0f}% data)")
    print(f"{'='*80}")

    # Temporarily update config
    update_trainer_config(trainer_script, ticker, data_fraction)

    # Run trainer
    try:
        result = subprocess.run(
            [sys.executable, str(trainer_script)],
            cwd=TRAINING_DIR,
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Training stopped by user")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Central training orchestration for LOB models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        nargs="+",
        default=["lit"],
        help="Models to train: baseline, deeplob, lit, tlob, or 'all'. Default: lit",
    )

    parser.add_argument(
        "--ticker",
        nargs="+",
        default=["CSCO"],
        help="Tickers to train on: CSCO, AAPL, GOOG, INTC, or 'all'. Default: CSCO",
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to use (0, 1]. Default: 0.1 (10%)",
    )

    args = parser.parse_args()

    # Validate and normalize
    models, tickers = validate_args(args)

    print("=" * 80)
    print("TRAINING ORCHESTRATION")
    print("=" * 80)
    print(f"Models:        {', '.join(models)}")
    print(f"Tickers:       {', '.join(tickers)}")
    print(f"Data fraction: {args.data_fraction:.2%}")
    print("=" * 80)

    # Run training combinations
    total_configs = len(models) * len(tickers)
    completed = 0
    failed = 0

    for model in models:
        for ticker in tickers:
            completed += 1
            print(f"\n[{completed}/{total_configs}]", end=" ")
            
            ret = run_trainer(model, ticker, args.data_fraction)
            if ret != 0:
                failed += 1
                print(f"FAILED (exit code {ret})")
            else:
                print(f"SUCCESS")

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_configs - failed}/{total_configs} configs completed")
    if failed > 0:
        print(f"WARNING: {failed} configs failed")
        sys.exit(1)
    else:
        print("All training runs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""Generate clearly labelled SYNTHETIC development data; never use it as benchmark evidence."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402


def generate_synthetic_transactions(rows: int, seed: int) -> pd.DataFrame:
    """Create a minimal ULB-shaped development sample with an explicit source label."""
    if rows < 100:
        raise ValueError("Use at least 100 rows so both classes can be represented.")
    rng = np.random.default_rng(seed)
    fraud = rng.random(rows) < 0.02
    frame = pd.DataFrame({
        "Time": rng.uniform(0, 172_800, rows),
        "Amount": rng.lognormal(mean=3.3 + fraud * 1.0, sigma=1.0, size=rows),
        "V1": rng.normal(loc=-1.2 * fraud, scale=1.0, size=rows),
        "V2": rng.normal(loc=1.0 * fraud, scale=1.0, size=rows),
        "V3": rng.normal(loc=-1.5 * fraud, scale=1.0, size=rows),
        "Class": fraud.astype("int8"),
        "data_source": "SYNTHETIC_DEVELOPMENT_ONLY",
    })
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/synthetic_transactions.csv")
    args = parser.parse_args()
    frame = generate_synthetic_transactions(args.rows, load_config().random_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("SYNTHETIC DATA ONLY — not real transaction data and not valid benchmark evidence.")
    print(f"Saved {len(frame)} rows to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


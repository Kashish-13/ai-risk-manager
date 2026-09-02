"""Download the public ULB/Worldline fraud benchmark from OpenML dataset 1597."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402


def download_openml_1597(output_path: Path) -> pd.DataFrame:
    """Fetch dataset 1597 and write a portable CSV; no credentials are required."""
    try:
        # Keep the OpenML cache inside the repository so the script works in
        # restricted environments and remains reproducible across machines.
        data_home = ROOT / "data" / "openml_cache"
        bunch = fetch_openml(
            data_id=1597, as_frame=True, parser="auto", data_home=str(data_home)
        )
    except Exception as exc:  # Network/API failures vary by environment.
        raise RuntimeError(
            "Unable to download OpenML dataset 1597. Check network access and retry, "
            "or manually place the ULB CSV in data/raw/. Original error: " + str(exc)
        ) from exc
    frame = bunch.frame.copy()
    if frame is None or frame.empty:
        raise RuntimeError("OpenML returned an empty dataset for id 1597.")
    target_name = getattr(bunch, "target_names", ["Class"])
    target_name = target_name[0] if isinstance(target_name, list) else target_name
    if "Class" not in frame.columns and target_name in frame.columns:
        frame = frame.rename(columns={target_name: "Class"})
    if "Class" not in frame.columns:
        raise RuntimeError(f"OpenML dataset has no expected target 'Class'. Columns: {frame.columns.tolist()}")
    frame["Class"] = pd.to_numeric(frame["Class"], errors="raise").astype("int8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="CSV destination (defaults to config primary_file).")
    parser.add_argument("--force", action="store_true", help="Refresh from OpenML even when a local CSV exists.")
    args = parser.parse_args()
    output = args.output or load_config().primary_file
    if output.exists() and not args.force:
        try:
            frame = pd.read_csv(output)
        except (OSError, pd.errors.ParserError) as exc:
            print(f"LOCAL DATASET READ FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"Using existing local OpenML dataset 1597 copy: {output}")
    else:
        try:
            frame = download_openml_1597(output)
        except RuntimeError as exc:
            print(f"DOWNLOAD FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"Saved OpenML dataset 1597 to: {output}")
    if "Class" not in frame.columns:
        print("DATASET ERROR: local file has no expected target column 'Class'.", file=sys.stderr)
        return 1
    print(f"Dataset shape: {frame.shape}")
    print(f"Columns: {frame.columns.tolist()}")
    print("Target distribution:")
    print(frame["Class"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

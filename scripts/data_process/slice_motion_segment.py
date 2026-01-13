import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir", required=True, help="Folder containing motion.npz and meta.json"
    )
    ap.add_argument(
        "--start", type=int, required=True, help="Start frame index (inclusive)"
    )
    ap.add_argument(
        "--end", type=int, required=True, help="End frame index (exclusive)"
    )
    ap.add_argument(
        "--output-dir", required=True, help="Target folder for sliced output"
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_path = input_dir / "motion.npz"
    meta_path = input_dir / "meta.json"
    if not motion_path.is_file():
        raise FileNotFoundError(f"Missing motion.npz in {input_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json in {input_dir}")

    if args.start < 0 or args.end <= args.start:
        raise ValueError("Expected 0 <= start < end")

    motion = dict(np.load(motion_path, allow_pickle=True))
    length = None
    for v in motion.values():
        if hasattr(v, "shape") and len(v.shape) >= 1:
            length = v.shape[0]
            break
    if length is None:
        raise ValueError("Unable to determine motion length")
    if args.end > length:
        raise ValueError(f"End index {args.end} exceeds motion length {length}")

    sliced = {}
    for k, v in motion.items():
        if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == length:
            sliced[k] = v[args.start : args.end]
        else:
            sliced[k] = v

    np.savez(output_dir / "motion.npz", **sliced)
    output_dir.joinpath("meta.json").write_text(meta_path.read_text())
    print(f"Saved slice: {output_dir}")


if __name__ == "__main__":
    main()

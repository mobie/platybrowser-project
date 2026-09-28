#!/usr/bin/env -S uv run --python 3.12
# /// script
# dependencies = ["numpy", "z5py"]
# ///
"""Zero the voxels of a mask N5 from another N5, level by level.

``out = in AND NOT mask`` at every pyramid level, mirroring the input's levels,
chunks and group/dataset attributes (uint8 + gzip + fillvalue 0).

Usage:
    ./subtract_mask.py --in-n5 <half.n5> --mask <foregut_lumen.n5> --out-n5 <out.n5>
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import z5py

from generate_shell_halves import (
    block_slices,
    mirror_group_attrs,
    mirror_level_info,
)


def subtract(in_path, mask_path, out_path, gzip_level=1):
    """Write ``in_path`` with every voxel nonzero in ``mask_path`` zeroed."""
    levels = mirror_level_info(in_path)
    group_attrs = mirror_group_attrs(in_path)
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    with z5py.File(str(out_path), "a") as f:
        setup = f.create_group("setup0")
        for k, v in group_attrs.get("setup0", {}).items():
            setup.attrs[k] = v
        tp = setup.create_group("timepoint0")
        for k, v in group_attrs.get("timepoint0", {}).items():
            tp.attrs[k] = v
        for lvl in levels:
            ds = tp.create_dataset(
                lvl["name"], shape=lvl["shape"], chunks=lvl["chunks"],
                dtype="uint8", compression="gzip", level=gzip_level,
                fillvalue=0,
            )
            for k, v in lvl["attrs"].items():
                ds.attrs[k] = v

    with z5py.File(str(in_path), "r") as sf, \
            z5py.File(str(mask_path), "r") as mf, \
            z5py.File(str(out_path), "a") as of:
        stp = sf["setup0/timepoint0"]
        mtp = mf["setup0/timepoint0"]
        otp = of["setup0/timepoint0"]
        for lvl in levels:
            sds = stp[lvl["name"]]
            mds = mtp[lvl["name"]]
            ods = otp[lvl["name"]]
            for sl in block_slices(lvl["shape"], lvl["chunks"]):
                block = sds[sl]
                if not block.any():
                    continue
                out = np.where(mds[sl] > 0, 0, block).astype(np.uint8)
                ods[sl] = out
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-n5", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out-n5", required=True)
    p.add_argument("--gzip-level", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    out = subtract(args.in_n5, args.mask, args.out_n5, gzip_level=args.gzip_level)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

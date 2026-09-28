#!/usr/bin/env -S uv run --python 3.12
# /// script
# dependencies = ["numpy", "z5py", "scipy"]
# ///
"""Build ``shell_foregut_lumen``: the near-axis foregut-lumen mass.

Selects shell voxels within ``--radius`` µm of one of two cylinders: the
segment ``--segment-p0`` → ``--segment-p1`` (µm) and its mirror across the
left/right symmetry plane. That selection also catches unrelated disjoint
lateral structures (parapodia / antennae: 3 small loops far from the sagittal
plane). With ``--keep-near-plane D`` the result is connected-component filtered:
only components whose median ``|x - y|`` is ``< D`` are kept, which leaves the
single large foregut-lumen mass that lies in the sagittal (x = y) plane.

The output mirrors the source pyramid (levels, shapes, chunks, group/dataset
attributes) with uint8 + gzip + fillvalue 0.

Usage:
    ./make_foregut_lumen.py \
        --shell-mask <shell.n5> \
        --out-n5 data/rawdata/shell_halves/shell_foregut_lumen.n5 \
        --local-xml-dir data/platybrowser_6dpf/images/local \
        --s3-xml-dir data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves \
        --segment-p0 157.0,152.2,47.1 --segment-p1 176.5,157.9,141.7 \
        --radius 25 --mirror --keep-near-plane 150
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import z5py
from scipy import ndimage

from generate_shell_halves import (
    _level_ds_factor,
    block_slices,
    mirror_group_attrs,
    mirror_level_info,
    shell_frame,
    write_local_xmls,
    write_s3_xmls,
)

NAME = "shell_foregut_lumen"
Z_CHUNK = 16


def build_segments(args, center, lr):
    """List of ``(p0_um, unit_direction, length_um)`` for the lung cylinders."""
    p0 = np.array([float(v) for v in args.segment_p0.split(",")])
    p1 = np.array([float(v) for v in args.segment_p1.split(",")])
    segments = []
    for a, b in ((p0, p1),):
        delta = b - a
        segments.append((a, delta / np.linalg.norm(delta),
                         float(np.linalg.norm(delta))))
    if args.mirror:
        def reflect(p):
            return p - 2.0 * float((p - center) @ lr) * lr
        a, b = reflect(p0), reflect(p1)
        delta = b - a
        segments.append((a, delta / np.linalg.norm(delta),
                         float(np.linalg.norm(delta))))
    return segments


def cylinder_mask_level(ds, ds_factor, segments, radius, resolution):
    """Full-level bool mask of shell voxels inside any cylinder."""
    shape = ds.shape
    out = np.zeros(shape, dtype=bool)
    off = (ds_factor - 1.0) / 2.0
    for z0 in range(0, shape[0], Z_CHUNK):
        blk = ds[z0:z0 + Z_CHUNK]
        if not blk.any():
            continue
        z = (np.arange(blk.shape[0]) + z0) * ds_factor + off
        y = np.arange(blk.shape[1]) * ds_factor + off
        x = np.arange(blk.shape[2]) * ds_factor + off
        zg, yg, xg = np.meshgrid(z, y, x, indexing="ij")
        points = np.stack([xg, yg, zg], axis=-1) * resolution
        keep = np.zeros(blk.shape, dtype=bool)
        for p0, direction, length in segments:
            t = (points - p0) @ direction
            perp = points - p0 - t[..., None] * direction
            keep |= ((t >= 0.0) & (t <= length)
                     & (np.linalg.norm(perp, axis=-1) < radius))
        out[z0:z0 + Z_CHUNK] = keep & (blk > 0)
    return out


def keep_components_near_plane(mask, max_abs_xy, ds_factor=1.0):
    """Zero components of ``mask`` whose median ``|x - y|`` is >= max_abs_xy.

    ``|x - y|`` is measured in full-resolution voxels: level indices are scaled
    by ``ds_factor`` so the same threshold applies at every pyramid level.
    """
    lbl, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    kept = 0
    for i in range(1, n + 1):
        zz, yy, xx = np.nonzero(lbl == i)
        if np.median(np.abs(xx - yy)) * ds_factor >= max_abs_xy:
            mask[lbl == i] = False
        else:
            kept += 1
    return kept, n


def make_lung_shell(mask_path, out_path, segments, radius, resolution,
                    keep_near_plane=None, gzip_level=1):
    """Write the (optionally component-filtered) lung cylinders as a new N5."""
    levels = mirror_level_info(mask_path)
    group_attrs = mirror_group_attrs(mask_path)
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

    with z5py.File(str(mask_path), "r") as sf, z5py.File(str(out_path), "a") as of:
        stp = sf["setup0/timepoint0"]
        otp = of["setup0/timepoint0"]
        for lvl in levels:
            ds_factor = _level_ds_factor(lvl)
            mask = cylinder_mask_level(stp[lvl["name"]], ds_factor, segments,
                                       radius, resolution)
            if keep_near_plane is not None and mask.any():
                kept, total = keep_components_near_plane(mask, keep_near_plane,
                                                         ds_factor)
                print(f"  {lvl['name']}: kept {kept}/{total} components")
            ods = otp[lvl["name"]]
            for sl in block_slices(lvl["shape"], lvl["chunks"]):
                if not mask[sl].any():
                    continue
                block = stp[lvl["name"]][sl]
                out = np.zeros_like(block)
                sel = mask[sl] & (block > 0)
                out[sel] = block[sel]
                ods[sl] = out
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shell-mask", required=True)
    p.add_argument("--out-n5", required=True)
    p.add_argument("--local-xml-dir", required=True)
    p.add_argument("--s3-xml-dir", required=True)
    p.add_argument("--segment-p0", required=True, help="'x,y,z' in µm.")
    p.add_argument("--segment-p1", required=True, help="'x,y,z' in µm.")
    p.add_argument("--radius", type=float, default=25.0, help="Cylinder radius, µm.")
    p.add_argument("--mirror", action="store_true",
                   help="Also keep the mirror of the segment across the symmetry plane.")
    p.add_argument("--keep-near-plane", type=float, default=None,
                   help="Keep only connected components whose median |x - y| is "
                        "below this many voxels (removes the lateral parapodia).")
    p.add_argument("--gzip-level", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    with z5py.File(str(args.shell_mask), "r") as f:
        resolution = np.asarray(
            f["setup0/timepoint0"].attrs["resolution"], dtype=float)
    frame = shell_frame(args.shell_mask)
    segments = build_segments(args, frame["center"], frame["lr"])
    print("cylinders (µm):")
    for p0, direction, length in segments:
        print("  {} -> {} (len {:.1f}, r {})".format(
            np.round(p0, 1), np.round(p0 + direction * length, 1), length,
            args.radius))
    out = make_lung_shell(args.shell_mask, args.out_n5, segments, args.radius,
                          resolution, keep_near_plane=args.keep_near_plane,
                          gzip_level=args.gzip_level)
    write_local_xmls([NAME], Path(args.local_xml_dir), Path(args.out_n5).parent)
    write_s3_xmls([NAME], Path(args.s3_xml_dir))
    print(f"Wrote {out} and {NAME}.xml (local + s3)")


if __name__ == "__main__":
    main()

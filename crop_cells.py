"""Crop CytoFormer input patches from a whole-slide image.

CytoFormer classifies one cell at a time from a fixed physical field of view centred on the
nucleus. This script takes a WSI and a table of nucleus centroids in that WSI's pixel frame and
writes one 224x224 PNG per cell, so the output folder can be passed straight to infer.py.

The window is PATCH_UM micrometres wide at the slide's native resolution (so its size in pixels is
PATCH_UM / mpp) and is then resized to 224 px. Edges are padded with white.

Input CSV must have columns: x, y   (nucleus centroid, in pixels of the WSI at level 0)
Optional column:            cell_id (defaults to the row index)

Usage:
  python crop_cells.py --wsi slide.ome.tif --cells cells.csv --mpp 0.25 --out patches/
  python crop_cells.py --wsi slide.ome.tif --cells cells.csv --mpp 0.25 --out patches/ --fov_um 56
"""
import os, argparse
import numpy as np, pandas as pd
import cv2, tifffile, zarr

OUT_PX = 224


def open_level0(wsi):
    """Return (zarr array, is_channels_first, shape). Reads the finest pyramid level lazily."""
    z = zarr.open(tifffile.imread(wsi, aszarr=True, level=0), mode="r")
    shape = z.shape
    chw = len(shape) == 3 and shape[0] <= 4 and shape[2] > 4      # (C,H,W) vs (H,W,C)
    return z, chw, shape


def crop_one(z, chw, shape, cx, cy, half):
    """Crop a 2*half window centred on (cx, cy), white-padded at the edges, resized to OUT_PX."""
    out = np.full((2 * half, 2 * half, 3), 255, np.uint8)
    H, W = (shape[1], shape[2]) if chw else (shape[0], shape[1])
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    if x1 > x0 and y1 > y0:
        if chw:
            win = np.asarray(z[:3, y0:y1, x0:x1]).transpose(1, 2, 0)
        else:
            win = np.asarray(z[y0:y1, x0:x1])
            if win.ndim == 2:
                win = np.stack([win] * 3, -1)
            win = win[..., :3]
        out[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)] = win
    if out.shape[0] != OUT_PX:
        out = cv2.resize(out, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", required=True, help="whole-slide image (pyramidal OME-TIFF/TIFF)")
    ap.add_argument("--cells", required=True, help="CSV with columns x, y (WSI pixels at level 0)")
    ap.add_argument("--mpp", type=float, required=True, help="micrometres per pixel of the WSI at level 0")
    ap.add_argument("--out", required=True, help="output folder for the patches")
    ap.add_argument("--fov_um", type=float, default=56.0, help="physical field of view, default 56 um")
    ap.add_argument("--limit", type=int, default=0, help="only crop the first N cells (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.cells)
    for c in ("x", "y"):
        assert c in df.columns, f"--cells must have a '{c}' column"
    if "cell_id" not in df.columns:
        df["cell_id"] = np.arange(len(df))
    if args.limit:
        df = df.head(args.limit)

    half = max(8, int(round(0.5 * args.fov_um / args.mpp)))
    os.makedirs(args.out, exist_ok=True)
    z, chw, shape = open_level0(args.wsi)
    print(f"WSI {shape}, {'CHW' if chw else 'HWC'} | {args.fov_um} um at {args.mpp} um/px "
          f"-> {2*half} px window -> {OUT_PX} px | {len(df):,} cells", flush=True)

    for i, (cid, x, y) in enumerate(zip(df.cell_id, df.x, df.y)):
        patch = crop_one(z, chw, shape, int(round(x)), int(round(y)), half)
        cv2.imwrite(os.path.join(args.out, f"{cid}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{len(df):,}", flush=True)
    print(f"wrote {len(df):,} patches to {args.out}", flush=True)


if __name__ == "__main__":
    main()

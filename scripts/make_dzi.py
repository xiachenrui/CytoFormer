"""Build a Deep Zoom (DZI) tile pyramid so a large image can be panned and zoomed in a browser.

Used to prepare the sample slides shown in the online viewer (docs/). One call per image; the
viewer overlays the prediction pyramid on the H&E pyramid.

Usage:
  python scripts/make_dzi.py --image slide_HE.jpg --out docs/slides/NCBI785_he
  python scripts/make_dzi.py --image slide_pred.jpg --out docs/slides/NCBI785_pred --max_side 12000
"""
import os, argparse, math
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DZI = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
       'Format="{fmt}" Overlap="{ov}" TileSize="{ts}">'
       '<Size Width="{w}" Height="{h}"/></Image>\n')


def build(img, out, tile=512, overlap=1, quality=80, fmt="jpeg"):
    """Write <out>.dzi and <out>_files/<level>/<col>_<row>.<ext>."""
    ext = "jpg" if fmt == "jpeg" else fmt
    w, h = img.size
    levels = int(math.ceil(math.log2(max(w, h)))) + 1
    os.makedirs(f"{out}_files", exist_ok=True)
    with open(f"{out}.dzi", "w") as f:
        f.write(DZI.format(fmt=ext, ov=overlap, ts=tile, w=w, h=h))

    for level in range(levels):
        scale = 2 ** (levels - 1 - level)
        lw, lh = max(1, math.ceil(w / scale)), max(1, math.ceil(h / scale))
        lvl_img = img if scale == 1 else img.resize((lw, lh), Image.LANCZOS)
        d = f"{out}_files/{level}"
        os.makedirs(d, exist_ok=True)
        for col in range(math.ceil(lw / tile)):
            for row in range(math.ceil(lh / tile)):
                x0, y0 = col * tile, row * tile
                box = (max(0, x0 - overlap), max(0, y0 - overlap),
                       min(lw, x0 + tile + overlap), min(lh, y0 + tile + overlap))
                lvl_img.crop(box).save(f"{d}/{col}_{row}.{ext}", quality=quality)
        if lvl_img is not img:
            lvl_img.close()
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. docs/slides/NCBI785_he")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--max_side", type=int, default=0, help="downscale so the long side is at most this")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    if args.max_side and max(img.size) > args.max_side:
        s = args.max_side / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = build(img, args.out, tile=args.tile, quality=args.quality)
    print(f"{args.image} {img.size} -> {args.out}.dzi ({n} levels)")


if __name__ == "__main__":
    main()

"""Turn raw benchmark pictures into scene pictures: 848 x 464, one scene ID each.

Input: a folder with one sub-folder per kind of picture, named
<place>_<kind> (e.g. city_real/, nature_ai/), and an optional sources.txt
with lines "<folder>/<file> | source | note".

Output, in --out:
  images/<scene>.png   cropped and resized pictures; scene = <place>_<kind>_<n>
  scenes.json          per scene: place, kind, original file, size, crop, source
  contact_sheet.jpg    every scene, to check the crops by eye

Each picture first loses any black bars at its edges, then is cropped around
its center to the 848:464 shape (the same shape as the repo's example, so
every scene makes 464 x 848 videos), then resized. --zoom FILE=FACTOR crops
tighter around the center, e.g. to cut off a logo in a corner.

Usage:
  python -m eval.prepare_images ../benchmark_raw --out eval/benchmark/full \
      --zoom indoor_game/some-picture.jpg=1.1
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 848, 464
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp")


def read_sources(path):
    """sources.txt -> {"<folder>/<file>": (source, note)}."""
    sources = {}
    if not os.path.exists(path):
        return sources
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            sources[parts[0]] = (parts[1] if len(parts) > 1 else "",
                                 parts[2] if len(parts) > 2 else "")
    return sources


def trim_black_bars(image, threshold=12):
    """Box (left, top, right, bottom) without near-black rows/columns at the edges."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    rows = gray.mean(axis=1) > threshold
    cols = gray.mean(axis=0) > threshold
    if not rows.any() or not cols.any():
        return (0, 0, image.width, image.height)
    top, bottom = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    left, right = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return (int(left), int(top), int(right), int(bottom))


def crop_box(box, zoom=1.0, aspect=WIDTH / HEIGHT):
    """Largest box of the given aspect centered in `box`, shrunk by `zoom`."""
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    if w / h > aspect:
        cw, ch = h * aspect, h
    else:
        cw, ch = w, w / aspect
    cw, ch = cw / zoom, ch / zoom
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return (round(cx - cw / 2), round(cy - ch / 2), round(cx + cw / 2), round(cy + ch / 2))


def prepare(raw_dir, out_dir, zooms=None):
    """Crop every raw picture; returns the list of scene records."""
    zooms = zooms or {}
    sources = read_sources(os.path.join(raw_dir, "sources.txt"))
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    scenes = []
    for folder in sorted(os.listdir(raw_dir)):
        if not os.path.isdir(os.path.join(raw_dir, folder)) or "_" not in folder:
            continue
        place, kind = folder.rsplit("_", 1)
        files = sorted(f for f in os.listdir(os.path.join(raw_dir, folder))
                       if f.lower().endswith(IMAGE_EXTENSIONS))
        for n, name in enumerate(files, start=1):
            rel = f"{folder}/{name}"
            image = Image.open(os.path.join(raw_dir, rel)).convert("RGB")
            zoom = zooms.get(rel, 1.0)
            box = crop_box(trim_black_bars(image), zoom)
            scene = f"{place}_{kind}_{n}"
            image.crop(box).resize((WIDTH, HEIGHT), Image.LANCZOS).save(
                os.path.join(out_dir, "images", scene + ".png"))
            source, note = sources.get(rel, ("", ""))
            scenes.append({
                "scene": scene,
                "place": place,
                "kind": kind,
                "original": rel,
                "original_size": list(image.size),
                "crop_box": list(box),
                "zoom": zoom,
                "upscaled": (box[2] - box[0]) < WIDTH,
                "source": source,
                "note": note,
            })
    with open(os.path.join(out_dir, "scenes.json"), "w") as f:
        json.dump(scenes, f, indent=1)
    return scenes


def contact_sheet(out_dir, scenes, cols=4, width=424):
    height = round(width * HEIGHT / WIDTH)
    rows = (len(scenes) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * (height + 20)), "white")
    for k, s in enumerate(scenes):
        im = Image.open(os.path.join(out_dir, "images", s["scene"] + ".png")).resize((width, height))
        x, y = (k % cols) * width, (k // cols) * (height + 20)
        sheet.paste(im, (x, y + 20))
        label = s["scene"] + (" (upscaled)" if s["upscaled"] else "")
        ImageDraw.Draw(sheet).text((x + 4, y + 4), label, fill="black")
    path = os.path.join(out_dir, "contact_sheet.jpg")
    sheet.save(path, quality=85)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw_dir", help="Folder with <place>_<kind>/ sub-folders.")
    parser.add_argument("--out", required=True, help="Output folder (e.g. eval/benchmark/full).")
    parser.add_argument("--zoom", action="append", default=[], metavar="FILE=FACTOR",
                        help="Crop tighter for one picture, e.g. city_game/a.jpg=1.1.")
    args = parser.parse_args()
    zooms = {}
    for item in args.zoom:
        rel, factor = item.rsplit("=", 1)
        zooms[rel] = float(factor)

    scenes = prepare(args.raw_dir, args.out, zooms)
    sheet = contact_sheet(args.out, scenes)
    missing = [s["original"] for s in scenes if not s["source"]]
    print(f"{len(scenes)} scenes -> {os.path.join(args.out, 'images')}")
    print(f"{sum(s['upscaled'] for s in scenes)} were smaller than {WIDTH}x{HEIGHT} and enlarged")
    if missing:
        print(f"no source in sources.txt for: {missing}")
    print(f"check the crops: {sheet}")


if __name__ == "__main__":
    main()

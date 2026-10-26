import json
import os

import numpy as np
from PIL import Image

from eval import prepare_images as prep


def test_trim_black_bars():
    # A 200x120 picture with 20-pixel black bars at the top and bottom.
    a = np.full((120, 200, 3), 128, dtype=np.uint8)
    a[:20] = 0
    a[100:] = 0
    assert prep.trim_black_bars(Image.fromarray(a)) == (0, 20, 200, 100)


def test_crop_box_shape_and_zoom():
    wide = prep.crop_box((0, 0, 2000, 500))   # too wide: crop the sides
    tall = prep.crop_box((0, 0, 1000, 1000))  # too tall: crop top and bottom
    for left, top, right, bottom in (wide, tall):
        assert abs((right - left) / (bottom - top) - prep.WIDTH / prep.HEIGHT) < 0.01
    assert wide[1] == 0 and wide[3] == 500
    assert tall[0] == 0 and tall[2] == 1000
    zoomed = prep.crop_box((0, 0, 1000, 1000), zoom=2.0)
    assert zoomed[2] - zoomed[0] == 500  # half the width, same center


def test_prepare_names_and_records(tmp_path):
    raw = tmp_path / "raw"
    for folder, n in (("city_real", 2), ("nature_ai", 1)):
        os.makedirs(raw / folder)
        for k in range(n):
            Image.new("RGB", (1200, 700), (k * 50, 100, 100)).save(raw / folder / f"p{k}.jpg")
    (raw / "sources.txt").write_text("# comment\ncity_real/p0.jpg | Unsplash | photo\n")
    out = tmp_path / "out"

    scenes = prep.prepare(str(raw), str(out), zooms={"city_real/p1.jpg": 1.2})
    assert [s["scene"] for s in scenes] == ["city_real_1", "city_real_2", "nature_ai_1"]
    assert scenes[0]["source"] == "Unsplash" and scenes[1]["source"] == ""
    assert scenes[1]["zoom"] == 1.2
    for s in scenes:
        assert Image.open(out / "images" / (s["scene"] + ".png")).size == (prep.WIDTH, prep.HEIGHT)
    assert json.loads((out / "scenes.json").read_text()) == scenes

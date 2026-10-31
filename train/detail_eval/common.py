"""Shared pieces of the detail eval (train/detail_eval/): appearance-tag vocabulary rules, size buckets, head crops.

Pure Python + PIL so both venvs can import it (~/comfy-venv renders and embeds, ~/ccip-venv tags and detects).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

# deploy/modal_prep.py BUCKETS (not imported: that module imports modal)
BUCKETS = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (768, 1344), (1344, 768)]
QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres"
NEGATIVE = ("worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, chromatic aberration, "
            "bad anatomy, bad hands, extra arms, extra fingers, missing fingers, text, watermark, signature, artist name, "
            "multiple views, split screen, comic, 2koma, 4koma")   # asym_prep eval negative

# appearance = what should survive a pose / scene change: hair, eyes, persistent body marks, worn items
LOOK_SLOTS = {"appearance:hair": "hair", "appearance:eyes": "eyes", "clothing:headwear": "accessory",
              "clothing:accents": "accessory", "clothing:eyewear": "accessory", "clothing:outfit": "clothing",
              "clothing:top": "clothing", "clothing:bottom": "clothing", "clothing:legwear": "clothing",
              "clothing:footwear": "clothing"}
ANATOMY_KEEP = re.compile(r"(^|_)(ears?|horns?|tail|wings?|halo|mole|scar|tattoo|freckles|fangs?|skin|facial_mark|"
                          r"whiskers|markings|eyebrows)($|_)")
# pose, framing, expression or state tags that sit in the slots above
STATE_TAGS = {"full_body", "upper_body", "cowboy_shot", "portrait", "closed_eyes", "half-closed_eyes", "half-closed_eye",
              "one_eye_closed", "wide-eyed", "narrowed_eyes", "rolling_eyes", "empty_eyes", "crazy_eyes", "sparkling_eyes",
              "blank_eyes", "constricted_pupils", "glowing_eyes", "glowing_eye", "eyes_visible_through_hair",
              "floating_hair", "hair_spread_out", "wet_hair", "hair_flip", "messy_hair", "hair_over_shoulder",
              "alternate_costume", "official_alternate_costume", "alternate_hairstyle", "official_alternate_hairstyle",
              "alternate_eye_color", "alternate_hair_length", "alternate_hair_color", "cosplay", "adapted_costume",
              "open_clothes", "bare_legs", "bare_arms", "barefoot", "breasts", "cleavage"}
STATE_PATTERN = re.compile(r"^(no_|unworn_|holding_|adjusting_|looking_|covered_|hand_|hands_)|_(removed|lift|pull|"
                           r"down|up|aside|on_head|around_neck|in_mouth)$|breast")


def look_category(tag: str, slot: str | None) -> str | None:
    """Detail family of a Danbooru general tag (taxonomy slot), or None if it is not a persistent appearance tag."""
    if tag in STATE_TAGS or STATE_PATTERN.search(tag):
        return None
    if slot in LOOK_SLOTS:
        return LOOK_SLOTS[slot]
    if slot == "appearance:anatomy" and ANATOMY_KEEP.search(tag):
        return "body"
    return None


def bucket_for(w: int, h: int) -> tuple[int, int]:
    return min(BUCKETS, key=lambda b: abs(math.log((w / h) / (b[0] / b[1]))))


def head_box(dets, size: tuple[int, int], margin: float = 0.15) -> tuple[int, int, int, int] | None:
    """Square box around the most confident detection (imgutils.detect format), grown by margin, clipped."""
    if not dets:
        return None
    (x0, y0, x1, y1), _, _ = max(dets, key=lambda d: d[2])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + margin)
    box = (max(0, round(cx - half)), max(0, round(cy - half)), min(size[0], round(cx + half)), min(size[1], round(cy + half)))
    return box if box[2] - box[0] >= 16 and box[3] - box[1] >= 16 else None


def load_manifest(set_dir: Path) -> dict:
    return json.loads((Path(set_dir) / "manifest.json").read_text(encoding="utf-8"))


def self_check() -> None:
    assert look_category("long_hair", "appearance:hair") == "hair"
    assert look_category("closed_eyes", "appearance:eyes") is None
    assert look_category("red_eyes", "appearance:eyes") == "eyes"
    assert look_category("cat_ears", "appearance:anatomy") == "body"
    assert look_category("mole_under_eye", "appearance:anatomy") == "body"
    assert look_category("large_breasts", "appearance:anatomy") is None
    assert look_category("navel", "appearance:anatomy") is None
    assert look_category("holding_umbrella", "clothing:accents") is None
    assert look_category("hair_ornament", "clothing:headwear") == "accessory"
    assert look_category("unworn_hat", "clothing:headwear") is None
    assert look_category("full_body", "clothing:outfit") is None
    assert look_category("serafuku", "clothing:outfit") == "clothing"
    assert look_category("smile", "pose:expression") is None
    assert bucket_for(1000, 1500) == (832, 1216) and bucket_for(1000, 1000) == (1024, 1024)
    assert head_box([], (100, 100)) is None
    assert head_box([((10, 10, 30, 40), "head", 0.5), ((50, 50, 90, 90), "head", 0.9)], (100, 100)) == (44, 44, 96, 96)
    assert head_box([((0, 0, 20, 20), "head", 0.9)], (100, 100)) == (0, 0, 23, 23)


if __name__ == "__main__":
    self_check()
    print("common self-check ok")

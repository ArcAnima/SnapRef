"""Combined pre1 + pre2 figures for one community post (Arca), same design as make_pre_figures.py.

    python make_pre_combined.py RENDERS_DIR REFS_DIR CHAR_RENDERS_DIR OUT_DIR

Every comparison puts the two builds side by side with the same reference, prompt and seed:
  hero_v2a_pre1_pre2.png      the mosaic hero, tag "PRE1 · PRE2"
  banner.jpg                  per reference: pre1 row, pre2 row (3 scenes each)
  multi_reference.jpg         two references of one character -> pre1 | pre2, three scenes
  reference_weights.jpg       Persica pair and Persica + fox blend, weight 0 / 0.3-0.4 / 0.6-1, pre1 row + pre2 row
  character_reference.jpg     mode char: v1a | pre1 | pre2 (the biggest change), same seed
  which_one.jpg               a short card: pre1 vs pre2 numbers (detail eval) and when to use which
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

import make_pre_figures as mf
from make_pre_figures import (FAINT, INK, LINE, MUTED, ORANGE, arrow, background, card, contain, fit, font, footer,
                              hazard_band, mono, ref_card)

TAG = "PRE1 · PRE2"


def title(c: Image.Image, W: int, M: int, subtitle: str, right: list[str]) -> None:
    d = ImageDraw.Draw(c)
    big = font(104, "Black")
    d.text((M, 54), "SnapRef", font=big, fill=INK)
    tw = d.textlength("SnapRef", font=big)
    ver = font(52, "Light")
    d.text((M + tw + 18, 100), "v2a", font=ver, fill=INK)
    vw = d.textlength("v2a", font=ver)
    bx = int(M + tw + 18 + vw + 16)
    hazard_band(c, bx, 104, 260, 54, TAG)
    d.text((M, 178), subtitle, font=font(32, "Regular"), fill=MUTED)
    mf.right_block(d, W, M, right, bx + 260 + 32)


def build_tag(d: ImageDraw.ImageDraw, x: int, y: int, name: str, sub: str = "") -> None:
    """Small 'PRE1' / 'PRE2' label on the left of a row."""
    f = font(22, "ExtraBold")
    w = d.textlength(name, font=f) + 24
    d.rounded_rectangle((x, y, x + w, y + 36), 8, outline=ORANGE if name == "PRE2" else MUTED, width=2)
    d.text((x + 12, y + 5), name, font=f, fill=ORANGE if name == "PRE2" else INK)
    if sub:
        d.text((x, y + 44), sub, font=mono(16), fill=MUTED)


def banner(R: Path, refs: Path, out: Path) -> None:
    caps = {"museum": "art museum", "market": "night market", "train": "train at sunset"}
    PH, REF_W, SCENE_W, GAP, M, DIV, LBL = 400, 280, 274, 16, 64, 40, 110
    HEADER, CAP, PAIR_GAP, ROW_GAP, FOOT = 250, 40, 14, 44, 140
    W = M * 2 + REF_W + DIV + LBL + 3 * SCENE_W + 2 * GAP
    n = len(mf.BANNER_CUTS["pre2"])
    H = HEADER + n * (2 * (PH + CAP) + PAIR_GAP) + (n - 1) * ROW_GAP + FOOT
    c = background(W, H)
    title(c, W, M, "One reference image. Any pose, any scene.",
          ["REFERENCE ADAPTER FOR ANIMA", "two preview builds · v1a stays recommended"])
    d = ImageDraw.Draw(c)
    d.line((M, HEADER - 22, W - M, HEADER - 22), fill=LINE, width=2)
    y = HEADER + 6
    for k in range(n):
        ref = mf.BANNER_CUTS["pre2"][k][0]
        block_h = 2 * (PH + CAP) + PAIR_GAP
        x = M
        card(c, fit(refs / f"{ref}.png", REF_W, block_h - CAP), x, y, outline=ORANGE)
        d.text((x, y + block_h - CAP + 12), "REFERENCE", font=font(17, "Bold"), fill=ORANGE)
        x += REF_W
        arrow(d, x + DIV // 2, y, y + block_h - CAP)
        x0 = x + DIV
        for j, build in enumerate(("pre1", "pre2")):
            yy = y + j * (PH + CAP + PAIR_GAP)
            build_tag(d, x0, yy + PH // 2 - 18, build.upper())
            x = x0 + LBL
            for scene, seed in dict(mf.BANNER_CUTS[build])[ref]:
                card(c, fit(R / f"{build}_{ref}_{scene}_s{seed}.png", SCENE_W, PH), x, yy, outline=ORANGE if build == "pre2" else None)
                if j == 1:
                    d.text((x, yy + PH + 10), caps[scene], font=font(20, "Medium"), fill=INK)
                x += SCENE_W + GAP
        y += block_h + ROW_GAP
    footer(c, W, H, M, "huggingface.co/0ppxnhximxr/snapref-v2a-pre1 · -pre2",
           "Scene-only prompts. No character name, appearance or artist tags.")
    c.convert("RGB").save(out / "banner.jpg", quality=93)
    print("banner", c.size)


def multi_reference(R: Path, refs: Path, out: Path) -> None:
    PH, REF_W, OUT_W, GAP, M, LBL = 400, 250, 280, 16, 64, 110
    HEADER, CAP, PAIR_GAP, FOOT = 250, 40, 14, 140
    W = M * 2 + 2 * REF_W + GAP + 60 + LBL + 3 * OUT_W + 2 * GAP
    H = HEADER + 2 * (PH + CAP) + PAIR_GAP + 30 + FOOT
    c = background(W, H)
    title(c, W, M, "New: several references of one character.",
          ["MULTI-REFERENCE", "image + image_2 (+ image_3) · same character"])
    d = ImageDraw.Draw(c)
    d.line((M, HEADER - 22, W - M, HEADER - 22), fill=LINE, width=2)
    y = HEADER + 6
    block_h = 2 * (PH + CAP) + PAIR_GAP - CAP
    x = M
    card(c, ref_card(refs / "persica_A.png", REF_W, block_h), x, y, outline=ORANGE)
    d.text((x, y + block_h + 12), "IMAGE · face, hair", font=font(17, "Bold"), fill=ORANGE)
    x += REF_W + GAP
    card(c, contain(refs / "persica_B.png", REF_W, block_h), x, y, outline=ORANGE)
    d.text((x, y + block_h + 12), "IMAGE_2 · full outfit", font=font(17, "Bold"), fill=ORANGE)
    x += REF_W
    arrow(d, x + 30, y, y + block_h)
    x0 = x + 60
    caps = {"lab": "in the lab", "walk": "walking, autumn", "desk": "late night at the desk"}
    for j, build in enumerate(("pre1", "pre2")):
        yy = y + j * (PH + CAP + PAIR_GAP)
        build_tag(d, x0, yy + PH // 2 - 18, build.upper())
        x = x0 + LBL
        for scene, seed in (("lab", 101), ("walk", 202), ("desk", 101)):
            card(c, fit(R / f"{build}_persica_AB_{scene}_s{seed}.png", OUT_W, PH), x, yy, outline=ORANGE if build == "pre2" else None)
            if j == 1:
                d.text((x, yy + PH + 10), caps[scene], font=font(20, "Medium"), fill=INK)
            x += OUT_W + GAP
    footer(c, W, H, M, "SnapRef node: image, image_2, image_3 · multi_layout separate",
           "Each reference is encoded on its own; the target reads all of them.")
    c.convert("RGB").save(out / "multi_reference.jpg", quality=93)
    print("multi_reference", c.size)


def weights(R: Path, refs: Path, out: Path) -> None:
    PH, REF_W, OUT_W, GAP, M, LBL = 360, 220, 250, 14, 64, 110
    HEADER, CAP, PAIR_GAP, SECTION, FOOT = 250, 40, 12, 60, 140
    W = M * 2 + 2 * REF_W + GAP + 60 + LBL + 3 * OUT_W + 2 * GAP
    block = 2 * (PH + CAP) + PAIR_GAP
    H = HEADER + 2 * (SECTION + block) + 30 + FOOT
    c = background(W, H)
    title(c, W, M, "Reference weights: how much each image counts.",
          ["weight_1 · weight_2 · weight_3", "log(w) on that reference's keys · no speed cost"])
    d = ImageDraw.Draw(c)
    d.line((M, HEADER - 22, W - M, HEADER - 22), fill=LINE, width=2)
    sections = [
        ("SAME CHARACTER  ·  story art + full-body art", refs / "persica_A.png", refs / "persica_B.png",
         lambda b, w: R / f"{b}_persica_w{w}_park_s101.png", (0.0, 0.4, 1.0)),
        ("TWO CHARACTERS  ·  mix one into the other", refs / "persica_A.png", refs / "char_fox.png",
         lambda b, w: R / f"{b}_blend_persica_fox_w{w}_s202.png", (0.0, 0.3, 0.6)),
    ]
    y = HEADER + 4
    for head, ra, rb, path_of, ws in sections:
        d.text((M, y), head, font=mono(19), fill=MUTED)
        y += SECTION - 22
        x = M
        card(c, ref_card(ra, REF_W, block - CAP), x, y, outline=ORANGE)
        d.text((x, y + block - CAP + 10), "IMAGE · weight 1", font=font(16, "Bold"), fill=ORANGE)
        x += REF_W + GAP
        card(c, ref_card(rb, REF_W, block - CAP), x, y, outline=ORANGE)
        d.text((x, y + block - CAP + 10), "IMAGE_2 · weight w", font=font(16, "Bold"), fill=ORANGE)
        x += REF_W
        arrow(d, x + 30, y, y + block - CAP)
        x0 = x + 60
        for j, build in enumerate(("pre1", "pre2")):
            yy = y + j * (PH + CAP + PAIR_GAP)
            build_tag(d, x0, yy + PH // 2 - 18, build.upper())
            x = x0 + LBL
            for w in ws:
                card(c, fit(path_of(build, w), OUT_W, PH), x, yy, outline=ORANGE if (build == "pre2" and w == ws[1]) else None)
                if j == 1:
                    label = "w = 0  · image only" if w == 0 else (f"w = {w:g}" + ("  · equal" if w == 1.0 else ""))
                    d.text((x, yy + PH + 10), label, font=font(19, "Medium"), fill=INK)
                x += OUT_W + GAP
        y += block + 22
    footer(c, W, H, M, "weight_2 = 0.4: mostly the first image, a little of the second",
           "0 skips a reference. Separate layout only.")
    c.convert("RGB").save(out / "reference_weights.jpg", quality=93)
    print("reference_weights", c.size)


def character_mode(C: Path, refs: Path, out: Path) -> None:
    PH, REF_W, OUT_W, GAP, M, DIV = 430, 290, 290, 16, 64, 40
    HEADER, CAP, ROW_GAP, FOOT = 270, 46, 30, 150
    rows = [("char_250c", "park"), ("char_4a8d", "park"), ("char_8767", "cafe")]
    seed = 907228049305750
    W = M * 2 + REF_W + DIV + 3 * OUT_W + 2 * GAP
    H = HEADER + len(rows) * (PH + CAP) + (len(rows) - 1) * ROW_GAP + FOOT
    c = background(W, H)
    title(c, W, M, "Character reference: the character, not the art style.",
          ["MODE CHAR · BIGGEST CHANGE IN v2a", "prompt sets the style · reference sets the character"])
    d = ImageDraw.Draw(c)
    hx = M + REF_W + DIV
    for k, (head, color) in enumerate((("v1a · char", MUTED), ("v2a-pre1 · char", INK), ("v2a-pre2 · char", ORANGE))):
        d.text((hx + k * (OUT_W + GAP), HEADER - 40), head, font=font(20, "Bold"), fill=color)
    d.text((M, HEADER - 40), "REFERENCE", font=font(20, "Bold"), fill=ORANGE)
    d.line((M, HEADER - 12, W - M, HEADER - 12), fill=LINE, width=2)
    y = HEADER + 6
    for ref, scene in rows:
        x = M
        card(c, contain(refs / f"{ref}.webp", REF_W, PH), x, y, outline=ORANGE)
        x += REF_W
        arrow(d, x + DIV // 2, y, y + PH)
        x += DIV
        for model in ("v1a", "pre1", "pre2"):
            card(c, fit(C / f"{model}_char_{ref}_{scene}_s{seed}.png", OUT_W, PH), x, y, outline=ORANGE if model == "pre2" else None)
            x += OUT_W + GAP
        cap = {"park": "full body, standing, smile, park, day", "cafe": "cowboy shot, sitting, cafe, holding cup"}[scene]
        d.text((M + REF_W + DIV, y + PH + 12), f"prompt: {cap}   ·   same seed in every column", font=mono(18), fill=MUTED)
        y += PH + CAP + ROW_GAP
    footer(c, W, H, M, "mode char: hair, eyes, outfit from the reference; style from the base model and your prompt",
           "Scene-only prompts. No character name or artist tags.")
    c.convert("RGB").save(out / "character_reference.jpg", quality=93)
    print("character_reference", c.size)


def which_one(out: Path) -> None:
    """pre1 vs pre2 at a glance: detail-eval numbers (v2a minus v1a, 64 cases x 2 seeds) and a recommendation."""
    W, H, M = 1500, 980, 64
    c = background(W, H)
    title(c, W, M, "Two preview builds of the same run. Which one?",
          ["v2a-pre1 = step 10,000 · v2a-pre2 = step 15,000", "both continue from v1a · neither replaces it"])
    d = ImageDraw.Draw(c)
    d.line((M, 228, W - M, 228), fill=LINE, width=2)
    cols = [M, M + 520, M + 820, M + 1120]
    y = 262
    heads = ["detail eval (difference to v1a)", "v2a-pre1", "v2a-pre2", "better is"]
    for x, h in zip(cols, heads):
        d.text((x, y), h, font=font(22, "Bold"), fill=ORANGE if h == "v2a-pre2" else INK)
    y += 50
    rows = [("appearance tags kept (hair, eyes, outfit)", "-0.014", "-0.015", "higher"),
            ("head-crop match, SigLIP2", "-0.018", "-0.005", "higher"),
            ("head-crop distance, CCIP", "+0.010", "+0.005", "lower"),
            ("head-crop distance, CCIP · real-art refs", "+0.019", "+0.010", "lower"),
            ("full-body distance, CCIP", "+0.008", "+0.007", "lower")]
    for name, a, b, better in rows:
        d.line((M, y - 12, W - M, y - 12), fill=(255, 255, 255, 22), width=1)
        d.text((cols[0], y), name, font=font(22, "Regular"), fill=INK)
        d.text((cols[1], y), a, font=mono(24), fill=MUTED)
        d.text((cols[2], y), b, font=mono(24), fill=ORANGE)
        d.text((cols[3], y), better, font=font(20, "Regular"), fill=MUTED)
        y += 54
    y += 16
    d.text((M, y), "seed noise is about ±0.005 on the face scores and ±0.04 on tags. 0 = same as v1a.", font=mono(18), fill=FAINT)
    y += 70
    tips = [("pre2", "closer to v1a on heads and faces; pick this one to try the new features."),
            ("pre1", "the earlier checkpoint; kept for comparison."),
            ("v1a", "still the recommended release for everyday use.")]
    for name, text in tips:
        build_tag(d, M, y, name.upper())
        d.text((M + 130, y + 4), text, font=font(26, "Regular"), fill=INK)
        y += 62
    footer(c, W, H, M, "New in both builds: multi-reference, reference weights, stronger mode char",
           "Same ComfyUI node for v1a, pre1 and pre2.")
    c.convert("RGB").save(out / "which_one.jpg", quality=93)
    print("which_one", c.size)


def main() -> int:
    R, refs, C, out = (Path(a) for a in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    banner(R, refs, out)
    multi_reference(R, refs, out)
    weights(R, refs, out)
    character_mode(C, refs, out)
    which_one(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

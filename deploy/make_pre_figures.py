"""SnapRef v2a pre-release figures: banner, multi-reference explainer, reference-weight explainer.

    python make_pre_figures.py RENDERS_DIR REFS_DIR OUT_DIR [pre1|pre2]
RENDERS_DIR: wjob.py output (<model>_<ref>_<scene>_s<seed>.png); REFS_DIR: the reference PNGs.

Design: same editorial grid as the v1 banners (title block, reference card | outputs, captions), new palette for the
pre-release line: graphite ground, a fluorescent-orange accent (the hero's instant-photo orange), a thin diagonal
"PRE-RELEASE" hazard band and a monospace build line, so it reads as a newer line that is not final.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

import os
FONT = os.environ.get("SNAPREF_FONT", str(Path.home() / "AppData/Local/Microsoft/Windows/Fonts/PretendardVariable.ttf"))   # Pretendard (OFL)
MONO = os.environ.get("SNAPREF_MONO", "C:/Windows/Fonts/consola.ttf")
BG_TOP, BG_BOTTOM = (22, 23, 26), (34, 33, 38)
INK, MUTED, FAINT = (240, 238, 233), (160, 156, 150), (92, 90, 96)
ORANGE = (255, 90, 31)
LINE = (255, 255, 255, 40)
SHADOW = (0, 0, 0, 170)


def font(size: int, weight: str = "Regular"):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


def mono(size: int):
    return ImageFont.truetype(MONO, size)


def fit(path: Path, w: int, h: int, centering=(0.5, 0.3)) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (240, 238, 233))
        bg.paste(im, mask=im.split()[3])
        im = bg
    return ImageOps.fit(im.convert("RGB"), (w, h), Image.LANCZOS, centering=centering)


def contain(path: Path, w: int, h: int, bg=(244, 242, 238)) -> Image.Image:
    """Whole image inside w x h (aspect kept) on a light card: for full-body references that a crop would cut."""
    im = fit(path, *Image.open(path).size, centering=(0.5, 0.5))   # same alpha flattening as fit()
    im = ImageOps.contain(im, (w, h), Image.LANCZOS)
    out = Image.new("RGB", (w, h), bg)
    out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return out


def ref_card(path: Path, w: int, h: int) -> Image.Image:
    """Half-body references (clearly wider than the card) are cropped like the outputs; full-body ones are shown whole,
    so a wide reference does not get light bars above and below it on the dark ground."""
    iw, ih = Image.open(path).size
    return fit(path, w, h, centering=(0.5, 0.25)) if iw / ih > 1.15 * w / h else contain(path, w, h)


# banner cuts per build, picked from the 2-seed candidate sheets (pre1 furcoat museum s202 drew the character twice)
BANNER_CUTS = {
    "pre2": [("char_fedora", [("museum", 202), ("market", 202), ("train", 101)]),
             ("char_furcoat", [("museum", 101), ("market", 101), ("train", 202)]),
             ("char_fox", [("museum", 101), ("market", 101), ("train", 101)])],
    "pre1": [("char_fedora", [("museum", 202), ("market", 202), ("train", 101)]),
             ("char_furcoat", [("museum", 101), ("market", 101), ("train", 202)]),
             ("char_fox", [("museum", 101), ("market", 202), ("train", 101)])],
}
# two-character weight blend, picked from the 6-pair x 4-weight x 2-seed sheets (reports/2026-10-31_multi-ref-same-character/
# blend_cand_*.jpg): Persica + the fox-eared character reads as a clean gradient (fox ears/tail at 0.3, pink/blonde hybrid at 0.6)
BLENDS = {"pre1": ("persica_fox", "persica_A", "char_fox", 202, (0.0, 0.3, 0.6))}   # seed 202: 0.6 is a real pink/blonde hybrid


def background(W: int, H: int) -> Image.Image:
    grad = Image.linear_gradient("L").resize((W, H))
    canvas = Image.composite(Image.new("RGBA", (W, H), BG_BOTTOM + (255,)), Image.new("RGBA", (W, H), BG_TOP + (255,)), grad)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((W * 0.55, -320, W * 1.25, 360), fill=ORANGE + (38,))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(140)))
    return canvas


def card(canvas: Image.Image, im: Image.Image, x: int, y: int, radius: int = 12, outline=None) -> None:
    shadow = Image.new("RGBA", (im.width + 60, im.height + 60), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((30, 36, im.width + 30, im.height + 36), radius, fill=SHADOW)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(16)), (x - 30, y - 30))
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius, fill=255)
    canvas.paste(im, (x, y), mask)
    if outline:
        ImageDraw.Draw(canvas).rounded_rectangle((x - 3, y - 3, x + im.width + 2, y + im.height + 2), radius + 3,
                                                 outline=outline, width=3)


def hazard_band(canvas: Image.Image, x0: int, y0: int, w: int, h: int, text: str) -> None:
    """Small diagonal-striped tag (orange / graphite) with a label: the 'not final' marker."""
    band = Image.new("RGBA", (w, h), ORANGE + (255,))
    d = ImageDraw.Draw(band)
    for x in range(-h, w + h, 22):
        d.polygon([(x, h), (x + 11, h), (x + 11 + h, 0), (x + h, 0)], fill=(28, 26, 26, 255))
    inner = Image.new("RGBA", (w - 2 * 6, h - 2 * 6), (28, 26, 26, 255))
    band.alpha_composite(inner, (6, 6))
    f = font(int(h * 0.42), "ExtraBold")
    tw = d.textlength(text, font=f)
    ImageDraw.Draw(band).text(((w - tw) / 2, h * 0.24), text, font=f, fill=ORANGE)
    canvas.alpha_composite(band, (x0, y0))


def title_block(canvas: Image.Image, W: int, margin: int, tag: str, subtitle: str, right: list[str]) -> None:
    d = ImageDraw.Draw(canvas)
    big = font(104, "Black")
    d.text((margin, 54), "SnapRef", font=big, fill=INK)
    tw = d.textlength("SnapRef", font=big)
    ver = font(52, "Light")
    d.text((margin + tw + 18, 100), "v2a", font=ver, fill=INK)
    vw = d.textlength("v2a", font=ver)
    bx = int(margin + tw + 18 + vw + 16)
    hazard_band(canvas, bx, 104, 168, 54, tag.upper())
    d.text((margin, 178), subtitle, font=font(32, "Regular"), fill=MUTED)
    right_block(d, W, margin, right, bx + 168 + 32)


def right_block(d: ImageDraw.ImageDraw, W: int, margin: int, lines: list[str], min_x: int, y0: int = 92, step: int = 32) -> None:
    """Right-aligned spec lines (first bold, rest mono). A line that would reach left of min_x (the end of the title
    and its tag) is wrapped at ' · ', so it never runs into the tag on narrow figures."""
    out = []
    for k, line in enumerate(lines):
        f = font(20, "Bold") if k == 0 else mono(19)
        cur = ""
        for part in line.split(" · "):
            cand = part if not cur else cur + " · " + part
            if cur and W - margin - d.textlength(cand, font=f) < min_x:
                out.append((cur, f, k))
                cur = part
            else:
                cur = cand
        out.append((cur, f, k))
    assert len(out) <= 3, f"header spec too long for this width: {out}"
    for i, (text, f, k) in enumerate(out):
        assert W - margin - d.textlength(text, font=f) >= min_x, f"cannot fit '{text}' right of the tag"
        d.text((W - margin - d.textlength(text, font=f), y0 + i * step), text, font=f, fill=INK if k == 0 else MUTED)


def footer(canvas: Image.Image, W: int, H: int, margin: int, url: str, note: str) -> None:
    d = ImageDraw.Draw(canvas)
    fy = H - 82
    d.line((margin, fy - 24, W - margin, fy - 24), fill=LINE, width=2)
    fl, fr = font(26, "Bold"), font(21, "Regular")
    if d.textlength(url, font=fl) + d.textlength(note, font=fr) + 40 <= W - 2 * margin:
        d.text((margin, fy), url, font=fl, fill=INK)
        d.text((W - margin - d.textlength(note, font=fr), fy + 4), note, font=fr, fill=MUTED)
    else:   # too long for one line: note goes under the main line instead of overlapping it
        d.text((margin, fy - 8), url, font=fl, fill=INK)
        d.text((margin, fy + 30), note, font=fr, fill=MUTED)


def arrow(d: ImageDraw.ImageDraw, cx: int, y0: int, y1: int, color=ORANGE) -> None:
    d.line((cx, y0 + 30, cx, y1 - 30), fill=LINE, width=2)
    my = (y0 + y1) // 2
    d.polygon([(cx - 8, my - 10), (cx + 6, my), (cx - 8, my + 10)], fill=color)


# ------------------------------------------------------------------ figures

def banner(R: Path, refs: Path, out: Path, tag: str) -> None:
    model = tag
    caps = {"museum": "art museum", "market": "night market", "train": "train at sunset"}
    rows = [(ref, [(sc, seed, caps[sc]) for sc, seed in cuts]) for ref, cuts in BANNER_CUTS[tag]]
    PH, REF_W, SCENE_W, GAP, M, DIV = 470, 330, 322, 18, 64, 40
    HEADER, CAP, ROW_GAP, FOOT = 250, 44, 34, 140
    W = M * 2 + REF_W + DIV + 3 * SCENE_W + 2 * GAP
    H = HEADER + len(rows) * (PH + CAP) + (len(rows) - 1) * ROW_GAP + FOOT
    c = background(W, H)
    title_block(c, W, M, tag, "One reference image. Any pose, any scene.",
                ["REFERENCE ADAPTER FOR ANIMA", f"build v2a-{tag} · not final · v1a stays recommended"])
    d = ImageDraw.Draw(c)
    d.line((M, HEADER - 22, W - M, HEADER - 22), fill=LINE, width=2)
    y = HEADER + 6
    for ref, scenes in rows:
        x = M
        card(c, fit(refs / f"{ref}.png", REF_W, PH), x, y, outline=ORANGE)
        d.text((x, y + PH + 12), "REFERENCE", font=font(17, "Bold"), fill=ORANGE)
        x += REF_W
        arrow(d, x + DIV // 2, y, y + PH)
        x += DIV
        for scene, seed, cap in scenes:
            card(c, fit(R / f"{model}_{ref}_{scene}_s{seed}.png", SCENE_W, PH), x, y)
            d.text((x, y + PH + 12), cap, font=font(21, "Medium"), fill=INK)
            x += SCENE_W + GAP
        y += PH + CAP + ROW_GAP
    footer(c, W, H, M, f"huggingface.co/0ppxnhximxr/snapref-v2a-{tag}", "Scene-only prompts. No character name, appearance or artist tags.")
    c.convert("RGB").save(out / "banner.jpg", quality=93)
    print("banner", c.size)


def multi_reference(R: Path, refs: Path, out: Path, tag: str) -> None:
    """Two references of one character -> one character. Row 1: upper-body art + full-body art -> 3 outputs.
    Row 2: single reference (story art only) vs both, same seed: the full-body reference adds the outfit."""
    model = tag
    PH, REF_W, OUT_W, GAP, M = 430, 270, 300, 18, 64
    HEADER, CAP, ROW_GAP, FOOT = 250, 46, 70, 140
    W = M * 2 + 2 * REF_W + GAP + 60 + 3 * OUT_W + 2 * GAP
    H = HEADER + 2 * (PH + CAP) + ROW_GAP + 40 + FOOT
    c = background(W, H)
    title_block(c, W, M, tag, "New: several references of one character.",
                ["MULTI-REFERENCE", "image + image_2 (+ image_3) · same character"])
    d = ImageDraw.Draw(c)
    d.line((M, HEADER - 22, W - M, HEADER - 22), fill=LINE, width=2)

    def ref_pair(y, label_a, label_b):
        x = M
        card(c, fit(refs / "persica_A.png", REF_W, PH, centering=(0.5, 0.2)), x, y, outline=ORANGE)
        d.text((x, y + PH + 12), label_a, font=font(17, "Bold"), fill=ORANGE)
        x += REF_W + GAP
        card(c, contain(refs / "persica_B.png", REF_W, PH), x, y, outline=ORANGE)
        d.text((x, y + PH + 12), label_b, font=font(17, "Bold"), fill=ORANGE)
        return x + REF_W
    y = HEADER + 6
    x = ref_pair(y, "IMAGE · face, hair", "IMAGE_2 · full outfit")
    arrow(d, x + 30, y, y + PH)
    x += 60
    for scene, seed, cap in (("lab", 101, "in the lab"), ("walk", 202, "walking, autumn"), ("desk", 101, "late night at the desk")):
        card(c, fit(R / f"{model}_persica_AB_{scene}_s{seed}.png", OUT_W, PH), x, y)
        d.text((x, y + PH + 12), cap, font=font(21, "Medium"), fill=INK)
        x += OUT_W + GAP
    y += PH + CAP + ROW_GAP
    d.text((M, y - 44), "WHAT IMAGE_2 ADDS  ·  same seed, same prompt: full body, walking, city street, autumn leaves", font=mono(19), fill=MUTED)
    # row 2: A only vs A + B, two seeds
    x = M
    for seed in (101, 202):
        for kind, label in (("A", "IMAGE only"), ("AB", "IMAGE + IMAGE_2")):
            name = f"{model}_persica_A_walk_s{seed}.png" if kind == "A" else f"{model}_persica_AB_walk_s{seed}.png"
            w = (W - 2 * M - 3 * GAP) // 4
            card(c, fit(R / name, w, PH), x, y, outline=ORANGE if kind == "AB" else None)
            d.text((x, y + PH + 12), label + f" · seed {seed}", font=font(19, "Medium"), fill=INK if kind == "AB" else MUTED)
            x += w + GAP
    footer(c, W, H, M, "SnapRef node: image, image_2, image_3 · multi_layout separate",
           "Each reference is encoded on its own; the target reads all of them.")
    c.convert("RGB").save(out / "multi_reference.jpg", quality=93)
    print("multi_reference", c.size)


def _ref_file(refs: Path, stem: str) -> Path:
    for ext in (".png", ".webp", ".jpg"):
        if (refs / f"{stem}{ext}").exists():
            return refs / f"{stem}{ext}"
    raise FileNotFoundError(stem)


def weights(R: Path, refs: Path, out: Path, tag: str, blend: tuple | None = None) -> None:
    """Row 1: same character, two references (Persica story art + PNC full body), weight_2 0 / 0.4 / 1.
    Row 2 (blend=(pair, ref_a, ref_b, seed, weights)): two DIFFERENT characters mixed by weight_2."""
    model = tag
    PH, REF_W, GAP, M = 400, 250, 16, 64
    HEADER, CAP, FOOT, SECTION = 250, 46, 140, 56
    cols, OUT_W = 3, 300
    n_rows = 2
    W = M * 2 + 2 * REF_W + GAP + 60 + cols * OUT_W + (cols - 1) * GAP
    H = HEADER + n_rows * (PH + CAP + SECTION) + 40 + FOOT
    c = background(W, H)
    title_block(c, W, M, tag, "Reference weights: how much each image counts.",
                ["weight_1 · weight_2 · weight_3", "log(w) on that reference's keys · no speed cost"])
    d = ImageDraw.Draw(c)
    d.line((M, HEADER - 22, W - M, HEADER - 22), fill=LINE, width=2)

    def row(y, title, ref_a, ref_b, lab_a, lab_b, cells, highlight):
        d.text((M, y), title, font=mono(19), fill=MUTED)
        y += SECTION - 20
        x = M
        card(c, ref_card(ref_a, REF_W, PH), x, y, outline=ORANGE)
        d.text((x, y + PH + 12), lab_a, font=font(17, "Bold"), fill=ORANGE)
        x += REF_W + GAP
        card(c, ref_card(ref_b, REF_W, PH), x, y, outline=ORANGE)
        d.text((x, y + PH + 12), lab_b, font=font(17, "Bold"), fill=ORANGE)
        x += REF_W
        arrow(d, x + 30, y, y + PH)
        x += 60
        for w, path, label in cells:
            card(c, fit(path, OUT_W, PH), x, y, outline=ORANGE if w == highlight else None)
            d.text((x, y + PH + 12), label, font=font(21, "Medium"), fill=INK)
            x += OUT_W + GAP
        return y + PH + CAP + 20

    y = HEADER + 4
    labels = {0.0: "w = 0  · image only", 0.4: "w = 0.4", 1.0: "w = 1  · both equal"}
    y = row(y, "SAME CHARACTER  ·  story art + full-body art", refs / "persica_A.png", refs / "persica_B.png",
            "IMAGE · weight 1", "IMAGE_2 · weight w",
            [(w, R / f"{model}_persica_w{w}_park_s101.png", labels[w]) for w in (0.0, 0.4, 1.0)], 0.4)
    if blend:
        pair, a, b, seed, ws = blend
        lab = lambda w: {0.0: "w = 0  · first character"}.get(w, f"w = {w:g}" + ("  · both equal" if w == 1.0 else ""))   # noqa: E731
        row(y, "TWO CHARACTERS  ·  mix one into the other", _ref_file(refs, a), _ref_file(refs, b),
            "IMAGE · weight 1", "IMAGE_2 · weight w",
            [(w, R / f"{model}_blend_{pair}_w{w}_s{seed}.png", lab(w)) for w in ws], ws[1])
    else:
        row(y, "SAME CHARACTER  ·  second seed", refs / "persica_A.png", refs / "persica_B.png", "IMAGE · weight 1",
            "IMAGE_2 · weight w", [(w, R / f"{model}_persica_w{w}_park_s202.png", labels[w]) for w in (0.0, 0.4, 1.0)], 0.4)
    footer(c, W, H, M, "weight_2 = 0.4: mostly the first image, a little of the second",
           "0 skips a reference. Separate layout only.")
    c.convert("RGB").save(out / "reference_weights.jpg", quality=93)
    print("reference_weights", c.size)


def character_mode(R: Path, refs: Path, out: Path, tag: str, user_ref: Path | None = None, user_out: Path | None = None) -> None:
    """The headline of v2a pre: mode char keeps the character and leaves the reference's art style behind.
    Row per reference: reference | v1a char | v2a char (same seed, same scene prompt). Optional first row = the user's
    own example (250c reference -> pre2 char output)."""
    PH, REF_W, OUT_W, GAP, M, DIV = 450, 300, 300, 18, 64, 40
    HEADER, CAP, ROW_GAP, FOOT = 270, 46, 34, 150
    rows = [("char_250c", "park", 907228049305750), ("char_4a8d", "park", 907228049305750), ("char_8767", "cafe", 907228049305750)]
    W = M * 2 + REF_W + DIV + 3 * OUT_W + 2 * GAP
    H = HEADER + len(rows) * (PH + CAP) + (len(rows) - 1) * ROW_GAP + FOOT
    c = background(W, H)
    title_block(c, W, M, tag, "Character reference: the character, not the art style.",
                ["MODE CHAR · BIGGEST CHANGE IN v2a", "prompt sets the style · reference sets the character"])
    d = ImageDraw.Draw(c)
    # column heads
    hx = M + REF_W + DIV
    for k, (head, color) in enumerate((("v1a · char", MUTED), (f"v2a-{tag} · char", ORANGE), (f"v2a-{tag} · cs (for contrast)", MUTED))):
        d.text((hx + k * (OUT_W + GAP), HEADER - 40), head, font=font(20, "Bold"), fill=color)
    d.text((M, HEADER - 40), "REFERENCE", font=font(20, "Bold"), fill=ORANGE)
    d.line((M, HEADER - 12, W - M, HEADER - 12), fill=LINE, width=2)
    y = HEADER + 6
    for ref, scene, seed in rows:
        x = M
        card(c, contain(refs / f"{ref}.webp", REF_W, PH), x, y, outline=ORANGE)
        x += REF_W
        arrow(d, x + DIV // 2, y, y + PH)
        x += DIV
        for model, mode in (("v1a", "char"), (tag, "char"), (tag, "cs")):
            path = R / f"{model}_{mode}_{ref}_{scene}_s{seed}.png"
            card(c, fit(path, OUT_W, PH), x, y, outline=ORANGE if (model, mode) == (tag, "char") else None)
            x += OUT_W + GAP
        caption = {"park": "full body, standing, smile, park, day", "cafe": "cowboy shot, sitting, cafe, holding cup"}[scene]
        d.text((M + REF_W + DIV, y + PH + 12), f"prompt: {caption}   ·   same seed in every column", font=mono(18), fill=MUTED)
        y += PH + CAP + ROW_GAP
    footer(c, W, H, M, "mode char: hair, eyes, outfit from the reference; style from the base model and your prompt",
           "Scene-only prompts. No character name or artist tags.")
    c.convert("RGB").save(out / "character_reference.jpg", quality=93)
    print("character_reference", c.size)


def char_styles(R: Path, refs: Path, out: Path, tag: str) -> None:
    """mode char + an art-style prompt: the reference keeps the character, the prompt picks the look."""
    PH, REF_W, OUT_W, GAP, M, DIV = 450, 300, 300, 18, 64, 40
    HEADER, CAP, FOOT = 270, 60, 150
    W = M * 2 + REF_W + DIV + 3 * OUT_W + 2 * GAP
    H = HEADER + PH + CAP + FOOT
    c = background(W, H)
    title_block(c, W, M, tag, "Same character, the style you ask for.",
                ["MODE CHAR + STYLE TAGS", "style tags in the prompt · no artist tags"])
    d = ImageDraw.Draw(c)
    d.line((M, HEADER - 22, W - M, HEADER - 22), fill=LINE, width=2)
    y = HEADER + 6
    x = M
    card(c, contain(refs / "char_250c.webp", REF_W, PH), x, y, outline=ORANGE)
    d.text((x, y + PH + 12), "REFERENCE", font=font(17, "Bold"), fill=ORANGE)
    x += REF_W
    arrow(d, x + DIV // 2, y, y + PH)
    x += DIV
    for style, cap in (("watercolor", "watercolor (medium)"), ("retro", "1990s (style), retro artstyle"), ("flat", "flat color, thick outlines")):
        card(c, fit(R / f"{tag}_char_char_250c_style_{style}_s101.png", OUT_W, PH), x, y)
        d.text((x, y + PH + 12), cap, font=mono(18), fill=INK)
        x += OUT_W + GAP
    footer(c, W, H, M, "prompt: <style tags>, full body, standing, smile, park, day", "mode char · same seed")
    c.convert("RGB").save(out / "character_styles.jpg", quality=93)
    print("character_styles", c.size)


def main() -> int:
    R, refs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else "pre2"
    out.mkdir(parents=True, exist_ok=True)
    banner(R, refs, out, tag)
    multi_reference(R, refs, out, tag)
    blend = BLENDS.get(tag)
    if blend and not (R / f"{tag}_blend_{blend[0]}_w{blend[4][1]}_s{blend[3]}.png").exists():
        blend = None
    weights(R, refs, out, tag, blend)
    if len(sys.argv) > 5:
        character_mode(Path(sys.argv[5]), refs, out, tag)
        if (Path(sys.argv[5]) / f"{tag}_char_char_250c_style_watercolor_s101.png").exists():
            char_styles(Path(sys.argv[5]), refs, out, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

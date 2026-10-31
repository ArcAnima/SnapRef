"""Contact sheet: per group a row  ref_A | ref_B | truth | noref | A | B | stitch | multi | multi_bias (all 384 px high)."""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw
out, dest = Path(sys.argv[1]), Path(sys.argv[2])
groups = sys.argv[3].split(",") if len(sys.argv) > 3 else [c["group"] for c in json.load(open(out / "cases.json"))]
cols = ["ref_A.jpg", "ref_B.jpg", "truth.jpg", "noref.png", "A.png", "B.png", "stitch.png", "multi.png", "multi_bias.png"]
H = 384; rows = []
for g in groups:
    d = out / g
    if not all((d / c).exists() for c in cols): continue
    ims = [Image.open(d / c).convert("RGB") for c in cols]
    ims = [im.resize((round(im.width * H / im.height), H)) for im in ims]
    row = Image.new("RGB", (sum(i.width for i in ims) + 6 * len(ims), H + 22), "white"); dr = ImageDraw.Draw(row); x = 0
    for c, im in zip(cols, ims):
        row.paste(im, (x, 22)); dr.text((x + 3, 4), f"{g} {c.split('.')[0]}", fill="black"); x += im.width + 6
    rows.append(row)
W = max(r.width for r in rows); sheet = Image.new("RGB", (W, sum(r.height for r in rows)), "white"); y = 0
for r in rows: sheet.paste(r, (0, y)); y += r.height
sheet.save(dest, quality=85); print(sheet.size)

"""scenes_v2: prompt-only fixes that cost no generation time (user: keep generating, speed first).
- negative += exposure tags; background beach/ocean/underwater -> a random non-water vocab background.
Rewrites manifest.jsonl and the two shard files in place (records not yet complete), keeps *.pre-negfix copies."""
import json, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "anima-ref-adapter-research/datagen"))
import vocab
run = Path.home() / "anima-ref-adapter-research/runs/scenes_v2"
EXTRA = ", nsfw, nude, underwear, panties, bra, swimsuit, bikini, cleavage, midriff, navel, huge breasts"
WATER = {"beach, ocean", "underwater"}
bgs = [(b, w) for b, w in vocab.BACKGROUNDS if b not in WATER]
rng = random.Random(20261031)
def fix(r):
    if not r["negative"].endswith(EXTRA):
        r["negative"] = r["negative"] + EXTRA
    s = r["scene"]
    if s.get("background") in WATER:
        old = s["background"]; new = rng.choices([b for b, _ in bgs], weights=[w for _, w in bgs])[0]
        assert r["prompt"].count(old) == 1, r["record_id"]
        r["prompt"] = r["prompt"].replace(old, new); s["background"] = new
    return r
for name in ("manifest.jsonl", "shard_0.jsonl", "shard_1.jsonl"):
    p = run / name; shutil.copy(p, p.with_name(name + ".pre-negfix"))
    rows = [fix(json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(name, len(rows), "water left", sum(r["scene"]["background"] in WATER for r in rows), "neg ok", all(r["negative"].endswith(EXTRA) for r in rows))

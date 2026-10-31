"""Build the frozen detail-eval set (tags only, no image inspection, no model scores).

    ~/comfy-venv/bin/python train/detail_eval/build_set.py --out ~/detail_eval/set --real 64 --synthetic 160 --seeds 2

synthetic: the 160 eval groups of asym_prep (32 held-out characters), every character once before second groups. Reference = the group's first image; truth = the
  group's image whose framing differs most (full body / cowboy shot first), its stripped prompt (no character, artist
  or appearance tags) is the render prompt. One case per group.
real: Danbooru2026 posts, rating g, solo, one character, one artist, short side >= 768, no layout / AI / minor tags,
  whose character is in none of: v1a's 2,367 synthetic characters (+ 32 held-out), the v2 pool (~/v2_pool/records.jsonl)
  and the v2_mix_prep records. 1girl or 1boy, no monochrome / sketch / chibi style tags. Per character one (reference, truth) pair of the same artist whose appearance tags overlap
  most (Jaccard >= 0.5, >= 6 shared tags), not parent/child of each other; one character per artist. Prompt = quality,
  safe, gender, solo, the truth post's scene tags. Images are range-read from the Danbooru tars (one CDN resolve per tar).
Writes OUT/manifest.json and OUT/<case>/{ref.png, truth.jpg}.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import NEGATIVE, QUALITY, bucket_for, look_category  # noqa: E402

HOME = Path.home()
DANBOORU = "https://huggingface.co/datasets/nyanko-devs/danbooru2026/resolve/main/original/data-{:04d}.tar"
MINOR_TAGS = {"loli", "shota", "child", "toddler", "baby", "kindergarten_uniform", "aged_down", "female_child",
              "male_child", "elementary_school", "randoseru", "child_on_child", "onee-shota", "oppai_loli"}
LAYOUT_TAGS = {"comic", "4koma", "2koma", "multiple_views", "split_screen", "character_sheet", "reference_sheet",
               "text-only_page", "variations"}
AI_TAGS = {"ai-generated", "ai-assisted", "ai_generated", "ai_assisted"}
# references whose colours or proportions cannot carry the details (tag check only)
NO_DETAIL_TAGS = {"monochrome", "greyscale", "sketch", "lineart", "chibi", "3d", "photorealistic", "pixel_art",
                  "partially_colored", "spot_color", "silhouette", "faceless", "cosplay_photo"}
NOT_ARTIST = {"banned_artist", "unknown_artist", "anonymous", "artist_request", "third-party_edit", "third-party_source"}
IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
SCENE_CATS = {"pose", "scenery"}
SCENE_SLOTS = {"equipment:handheld", "creature:animal"}
DROP_TAGS = {"solo", "1girl", "1boy", "1other", "male_focus", "female_focus", "highres", "absurdres"}
FRAMING_RANK = {"full body": 0, "full body, wide shot": 1, "cowboy shot": 2, "upper body": 3, "portrait": 4}


def norm(name: str) -> str:
    return name.replace("\\(", "(").replace("\\)", ")").replace(" ", "_").lower()


def anima_tag(tag: str) -> str:   # datagen/catalog.anima_tag
    text = " ".join(tag.replace("_", " ").split()).replace("\\", "\\\\")
    return text.replace("(", "\\(").replace(")", "\\)")


def synthetic_cases(prep: Path, data_dir: Path, limit: int) -> list[dict]:
    ix = json.loads((prep / "index.json").read_text(encoding="utf-8"))
    cases = []
    for gid, g in sorted(ix["groups"].items()):
        if g["split"] != "eval":
            continue
        ref_id, others = g["record_ids"][0], g["record_ids"][1:]
        ref_fr = ix["records"][ref_id]["framing"]
        # a different framing than the reference, the most of the body visible first
        tgt_id = min(others, key=lambda r: (ix["records"][r]["framing"] == ref_fr, FRAMING_RANK.get(ix["records"][r]["framing"], 9), r))
        t = ix["records"][tgt_id]
        cases.append({"name": f"syn_{gid}", "kind": "synthetic", "character": g["character"],
                      "ref_src": str(data_dir / ix["records"][ref_id]["image"]), "truth_src": str(data_dir / t["image"]),
                      "prompt": ix["prompts"][t["prompts"]["stripped"]], "negative": NEGATIVE,
                      "width": t["width"], "height": t["height"], "ref_record": ref_id, "truth_record": tgt_id})
    # every held-out character once before any second group of a character
    seen = collections.Counter()
    for c in cases:
        c["_rank"] = seen[c["character"]]
        seen[c["character"]] += 1
    cases = sorted(cases, key=lambda c: (c["_rank"], c["name"]))[:limit]
    for c in cases:
        del c["_rank"]
    return cases


def excluded_characters() -> set[str]:
    used = json.loads((HOME / "gelbooru2026" / "characters_used.json").read_text(encoding="utf-8"))
    out = {norm(c) for c in used["train_characters"] + used["held_out"]}
    asym = json.loads((HOME / "hf_up" / "preps" / "asym_prep" / "index.json").read_text(encoding="utf-8"))
    out |= {norm(g["character"]) for g in asym["groups"].values()}
    with (HOME / "v2_pool" / "records.jsonl").open(encoding="utf-8") as f:
        out |= {norm(json.loads(line)["character"]) for line in f}
    mix = json.loads((HOME / "hf_up" / "preps" / "v2_mix_prep" / "index.json").read_text(encoding="utf-8"))
    out |= {norm(r["character"]) for r in mix["records"].values() if r.get("character")}
    return out


def real_candidates(danbooru: Path, tax: dict, excluded: set[str], seed: int) -> list[dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(danbooru / "metadata" / "posts-snapshot.parquet")
    cols = ["id", "rating", "image_width", "image_height", "tag_string_general", "tag_string_character", "tag_string_meta",
            "tag_string_artist", "file_ext", "is_deleted", "is_banned", "score", "parent_id"]
    groups = collections.defaultdict(list)
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(t["id"])):
            if t["rating"][i] != "g" or (t["file_ext"][i] or "").lower() not in IMAGE_EXT or t["is_deleted"][i] or t["is_banned"][i]:
                continue
            chars = (t["tag_string_character"][i] or "").split()
            artists = [a for a in (t["tag_string_artist"][i] or "").split() if a not in NOT_ARTIST]
            if len(chars) != 1 or len(artists) != 1 or chars[0] in excluded:
                continue
            general = (t["tag_string_general"][i] or "").split()
            tags = set(general) | set((t["tag_string_meta"][i] or "").split())
            if "solo" not in tags or tags & (LAYOUT_TAGS | AI_TAGS | MINOR_TAGS | NO_DETAIL_TAGS):
                continue
            if not ({"1girl", "1boy"} & tags) or {"1girl", "1boy"} <= tags:
                continue
            if min(t["image_width"][i] or 0, t["image_height"][i] or 0) < 768:
                continue
            look = frozenset(x for x in general if look_category(x, tax.get(x, (None, None))[1]))
            if len(look) < 6:
                continue
            gender = "1boy" if "1boy" in tags else "1girl"
            scene = [x for x in general if x not in DROP_TAGS and (tax.get(x, (None, None))[0] in SCENE_CATS
                                                                   or tax.get(x, (None, None))[1] in SCENE_SLOTS)]
            groups[(artists[0], chars[0])].append({"id": t["id"][i], "w": t["image_width"][i], "h": t["image_height"][i],
                                                   "score": t["score"][i] or 0, "parent": t["parent_id"][i],
                                                   "look": look, "scene": scene, "gender": gender})
        print(f"scan {g + 1}/{pf.metadata.num_row_groups}: {len(groups)} groups", file=sys.stderr, flush=True)
    best_per_char = {}
    for (artist, char), posts in groups.items():
        if len(posts) < 2:
            continue
        posts = sorted(posts, key=lambda p: -p["score"])[:12]
        best = None
        for a in range(len(posts)):
            for b in range(len(posts)):
                p, q = posts[a], posts[b]
                if a == b or p["parent"] == q["id"] or q["parent"] == p["id"] or (p["parent"] and p["parent"] == q["parent"]):
                    continue
                if p["gender"] != q["gender"] or not q["scene"]:
                    continue
                inter = len(p["look"] & q["look"])
                jac = inter / len(p["look"] | q["look"])
                if inter >= 6 and jac >= 0.5:
                    key = (jac, p["score"] + q["score"], -a, -b)   # ref = a, truth = b
                    if best is None or key > best[0]:
                        best = (key, p, q)
        if best and (char not in best_per_char or best[0] > best_per_char[char][0]):
            best_per_char[char] = (best[0], artist, best[1], best[2])
    rng = random.Random(seed)
    items = sorted(best_per_char.items())
    rng.shuffle(items)
    out, artists_used = [], set()
    for char, (key, artist, ref, truth) in items:
        if artist in artists_used:
            continue
        artists_used.add(artist)
        w, h = bucket_for(truth["w"], truth["h"])
        out.append({"name": f"real_{truth['id']}", "kind": "real", "character": char, "artist": artist,
                    "ref_post": ref["id"], "truth_post": truth["id"], "look_jaccard": round(key[0], 3),
                    "shared_look": sorted(ref["look"] & truth["look"]),
                    "prompt": ", ".join([QUALITY, "safe", ref["gender"], "solo"] + [anima_tag(x) for x in truth["scene"]]),
                    "negative": NEGATIVE, "width": w, "height": h})
    return out


def resolve_cdn(tar: int, token: str) -> str:
    """deploy/modal_prep.resolve_cdn: signed CDN URL of one tar (one resolve call per tar)."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    delay = 5.0
    for _ in range(10):
        try:
            opener.open(urllib.request.Request(DANBOORU.format(tar), method="HEAD",
                                               headers={"Authorization": f"Bearer {token}"}), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 307, 308) and e.headers.get("Location"):
                return e.headers["Location"]
            if e.code != 429 and e.code < 500:
                raise
        except Exception:
            pass
        time.sleep(delay)
        delay = min(120.0, delay * 2)
    raise RuntimeError(f"could not resolve tar {tar}")


def fetch(ids: list[int], danbooru: Path, token: str) -> dict[int, bytes]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(danbooru / "index" / "tar-index.parquet")
    t = t.filter(pc.is_in(t["id"], value_set=__import__("pyarrow").array(ids))).to_pylist()
    urls = {tar: resolve_cdn(tar, token) for tar in sorted({r["shard"] for r in t})}
    print(f"resolved {len(urls)} tars for {len(t)} posts", file=sys.stderr, flush=True)

    def get(r):
        for attempt in range(6):
            try:
                req = urllib.request.Request(urls[r["shard"]], headers={
                    "Range": f"bytes={r['offset_bytes']}-{r['offset_bytes'] + r['length_bytes'] - 1}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                if len(data) == r["length_bytes"]:
                    return r["id"], data
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    urls[r["shard"]] = resolve_cdn(r["shard"], token)
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
        return r["id"], None
    with ThreadPoolExecutor(16) as ex:
        return {pid: d for pid, d in ex.map(get, t) if d}


def save_pair(case_dir: Path, ref, truth) -> None:
    from PIL import Image
    case_dir.mkdir(parents=True, exist_ok=True)
    ref = ref.convert("RGB")
    s = min(1.0, (1024 * 1024 / (ref.width * ref.height)) ** 0.5)   # eval_pilot.load_ref caps at 1 MP anyway
    ref.resize((round(ref.width * s), round(ref.height * s)), Image.LANCZOS).save(case_dir / "ref.png")
    truth = truth.convert("RGB")
    s = min(1.0, 1216 / max(truth.size))
    truth.resize((round(truth.width * s), round(truth.height * s)), Image.LANCZOS).save(case_dir / "truth.jpg", quality=95)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synthetic", type=int, default=160)
    ap.add_argument("--real", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--prep", type=Path, default=HOME / "hf_up" / "preps" / "asym_prep")
    ap.add_argument("--data-dir", type=Path, default=HOME / "data" / "scenes3x4")
    ap.add_argument("--danbooru", type=Path, default=HOME / "danbooru2026")
    ap.add_argument("--candidates", type=Path, help="reuse a real_candidates.json instead of scanning")
    args = ap.parse_args()
    from PIL import Image
    if (args.out / "manifest.json").exists():
        raise SystemExit(f"{args.out}/manifest.json exists: the set is frozen")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = synthetic_cases(args.prep, args.data_dir, args.synthetic)
    for c in cases:
        save_pair(args.out / c["name"], Image.open(c["ref_src"]), Image.open(c["truth_src"]))
    if args.real:
        cand_path = args.candidates or args.out / "real_candidates.json"
        if not cand_path.exists():
            import pyarrow.parquet as pq
            t = pq.read_table(HOME / "hf" / "taxonomy" / "tags.parquet", columns=["tag", "danbooru_category", "category", "slot"]).to_pydict()
            tax = {tag: (cat, slot) for tag, dc, cat, slot in zip(t["tag"], t["danbooru_category"], t["category"], t["slot"]) if dc == "general"}
            cand_path.write_text(json.dumps(real_candidates(args.danbooru, tax, excluded_characters(), args.seed), indent=1), encoding="utf-8")
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
        print(f"{len(cand)} real candidates", file=sys.stderr, flush=True)
        # fetch a margin of candidates; undecodable posts are skipped in candidate order
        pool = cand[: int(args.real * 1.3) + 8]
        data = fetch([x for c in pool for x in (c["ref_post"], c["truth_post"])], args.danbooru, os.environ["HF_TOKEN"])
        real = []
        for c in pool:
            if len(real) == args.real:
                break
            try:
                ref, truth = (Image.open(BytesIO(data[c[k]])) for k in ("ref_post", "truth_post"))
                ref.load(), truth.load()
            except Exception as e:   # missing from the tar index or not decodable
                print(f"skip {c['name']}: {type(e).__name__}", file=sys.stderr)
                continue
            save_pair(args.out / c["name"], ref, truth)
            real.append(c)
        cases += real
    for k, c in enumerate(cases):
        c["seeds"] = [9000 + 10 * k + s for s in range(args.seeds)]
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed, "steps": 28, "cfg": 4.5, "mode": "cs",
                "n_synthetic": sum(c["kind"] == "synthetic" for c in cases), "n_real": sum(c["kind"] == "real" for c in cases),
                "cases": cases}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_synthetic", "n_real")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Style-reference baseline renders through ComfyUI's API (local, 127.0.0.1).

    ~/comfy-venv/bin/python train/style_baseline.py ~/style_data/style_eval/cases.json OUT_DIR [--adapter snapref_v1a.safetensors]

Per held-out artist case: the reference post x 2 scene prompts x 1 seed, for
  base   Anima without reference;
  cs     SnapRef mode cs (character and style from the reference), strength 0;
  cs_s1  SnapRef mode cs, strength 1 (stronger reference).
The prompt carries no artist tag, so any style in the output can only come from the reference.
Images: OUT_DIR/<artist>/<method>_<scene>.png; OUT_DIR/renders.json lists them.
"""
import argparse
import copy
import json
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8188"
HERE = Path(__file__).resolve().parent
Q = "masterpiece, best quality, score_7, newest, highres, absurdres, safe, "


def post(w):
    req = urllib.request.Request(URL + "/prompt", data=json.dumps({"prompt": w}).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["prompt_id"]


def wait(pid):
    while True:
        h = json.loads(urllib.request.urlopen(f"{URL}/history/{pid}").read())
        if pid in h:
            return h[pid]
        time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--adapter", default="snapref_v1a.safetensors")
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    d = json.loads(args.cases.read_text(encoding="utf-8"))
    base = json.loads((HERE.parent / "runs" / "inference_api" / "01_ours_anima_ref_kv.api.json").read_text())
    renders = []
    t0 = time.time()
    for c in d["cases"]:
        ref = c["ref"]
        for si, scene in enumerate(d["scenes"][: args.scenes]):
            for method in ("base", "cs", "cs_s1"):
                w = copy.deepcopy(base)
                prefix = f"style_baseline/{c['artist']}/{method}_s{si}"
                for n in w.values():
                    ct, i = n["class_type"], n["inputs"]
                    if ct == "LoadImage":
                        i["image"] = "style/" + ref["path"].split("style_eval/", 1)[1]
                    if ct == "SnapRef":
                        i.update(adapter=args.adapter, mode="cs", strength=1.0 if method == "cs_s1" else 0.0)
                    if ct == "CLIPTextEncode" and "masterpiece" in i["text"]:
                        i["text"] = Q + ref["gender"] + ", " + scene
                    if ct == "KSampler":
                        i.update(seed=args.seed, sampler_name="er_sde", steps=28)
                    if ct == "SaveImage":
                        i["filename_prefix"] = prefix
                if method == "base":   # KSampler takes the model straight from the loader
                    sr = next(k for k, n in w.items() if n["class_type"] == "SnapRef")
                    for n in w.values():
                        for key, v in n["inputs"].items():
                            if isinstance(v, list) and v and v[0] == sr:
                                n["inputs"][key] = list(w[sr]["inputs"]["model"])
                    del w[sr]
                h = wait(post(w))
                imgs = [im for o in h["outputs"].values() for im in o.get("images", [])]
                renders.append({"artist": c["artist"], "method": method, "scene": si, "ok": h["status"]["status_str"] == "success",
                                "file": f"{imgs[0]['subfolder']}/{imgs[0]['filename']}" if imgs else None})
        print(f"{c['artist']} done ({time.time() - t0:.0f}s)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "renders.json").write_text(json.dumps(renders, indent=1), encoding="utf-8")
    print(json.dumps({"renders": len(renders), "failed": sum(not r["ok"] for r in renders)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

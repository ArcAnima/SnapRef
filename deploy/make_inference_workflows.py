"""Workflows for the inference box (ComfyUI UI format + API format for testing).

    python deploy/make_inference_workflows.py OUT_DIR [--api-dir DIR]

  01_ours_anima_ref_kv.json     SnapRef node + KSampler
  02_electrongirl_v2.json       ElectronGirl v2 one-node (Anima Ref Adapter) + KSampler
  03_electrongirl_pilot.json    ElectronGirl pilot_r32 (Loader + Ref Latent + Guider + SamplerCustomAdvanced)
  04_role_ref_kv.json           AnticPan Role-Ref-KV (its own sampling node)
  05_compare_all.json           one reference and one prompt -> SnapRef | v2 | Role-Ref-KV | base, same seed
Shared defaults: Anima base v1.0 bf16, 832x1216, 28 steps, cfg 4.5, euler/simple (ours, base), er_sde (EG default).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POS = "masterpiece, best quality, score_7, safe, 1girl, solo, full body, standing, looking at viewer, smile, park, day"
NEG = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
SEED = 20261029
OURS = "snapref_v1a.safetensors"

# output types per node class (slot order), for links
OUT = {"UNETLoader": ["MODEL"], "CLIPLoader": ["CLIP"], "VAELoader": ["VAE"], "LoadImage": ["IMAGE", "MASK"],
       "CLIPTextEncode": ["CONDITIONING"], "EmptyLatentImage": ["LATENT"], "KSampler": ["LATENT"],
       "VAEDecode": ["IMAGE"], "SaveImage": [], "SnapRef": ["MODEL"], "AnimaRefAdapter": ["MODEL"],
       "AnimaRefAdapterLoader": ["MODEL"], "AnimaRefLatent": ["LATENT", "LATENT", "IMAGE"], "AnimaRefGuider": ["GUIDER"],
       "KSamplerSelect": ["SAMPLER"], "BasicScheduler": ["SIGMAS"], "RandomNoise": ["NOISE"],
       "SamplerCustomAdvanced": ["LATENT", "LATENT"], "AnimaRoleRefKV": ["LATENT"], "Note": [], "PrimitiveInt": ["INT"], "ImageStitch": ["IMAGE"],
       "SnapRefSimple": ["MODEL"]}
# widget names in order (UI widgets_values; KSampler-style seeds get "control_after_generate" after them)
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"], "CLIPLoader": ["clip_name", "type", "device"], "VAELoader": ["vae_name"],
    "LoadImage": ["image", "upload"], "CLIPTextEncode": ["text"], "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "@control", "steps", "cfg", "sampler_name", "scheduler", "denoise"], "VAEDecode": [],
    "SaveImage": ["filename_prefix"],
    "SnapRef": ["adapter", "mode", "ref_in_uncond", "ref_megapixels", "strength", "ref_caption", "ref_guidance_on", "ref_guidance",
                   "correction", "ref_end", "use_batch", "multi_layout", "weight_1", "weight_2", "weight_3"],
    "AnimaRefAdapter": ["lora_name", "ref_guidance", "correction", "strength", "ref_end", "ref_megapixels"],
    "AnimaRefAdapterLoader": ["lora_name", "strength", "ref_offset_h"],
    "AnimaRefLatent": ["width", "height", "fill"],
    "AnimaRefGuider": ["cfg", "ref_guidance", "correction", "rescale_phi", "ref_start", "ref_end", "outside"],
    "KSamplerSelect": ["sampler_name"], "BasicScheduler": ["scheduler", "steps", "denoise"],
    "RandomNoise": ["noise_seed", "@control"], "SamplerCustomAdvanced": [],
    "AnimaRoleRefKV": ["adapter", "seed", "@control", "width", "height", "steps", "cfg"], "Note": ["text"],
    "PrimitiveInt": ["value", "@control"],
    "ImageStitch": ["direction", "match_image_size", "spacing_width", "spacing_color"],
    "SnapRefSimple": ["adapter", "reference", "mode", "strength", "fidelity", "reference_2", "strength_2", "upload", "snapref_panel"],
}


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []

    def add(self, cls, inputs=None, widgets=None, pos=(0, 0), title=None):
        """inputs: {name: (node_id, slot) | ("type", name) placeholder}; widgets: {name: value}."""
        nid = len(self.nodes) + 1
        self.nodes.append({"id": nid, "cls": cls, "inputs": inputs or {}, "widgets": widgets or {}, "pos": pos, "title": title})
        return nid

    def ui(self) -> dict:
        nodes, links, out_links = [], [], {}
        for n in self.nodes:
            for name, (src, slot) in n["inputs"].items():
                lid = len(links) + 1
                links.append([lid, src, slot, n["id"], name, OUT[self.nodes[src - 1]["cls"]][slot]])
                out_links.setdefault((src, slot), []).append(lid)
        for n in self.nodes:
            ins = []
            for i, (name, (src, slot)) in enumerate(n["inputs"].items()):
                lid = next(l[0] for l in links if l[3] == n["id"] and l[4] == name)
                ins.append({"name": name, "type": OUT[self.nodes[src - 1]["cls"]][slot], "link": lid})
            outs = [{"name": t, "type": t, "links": out_links.get((n["id"], s), []), "slot_index": s}
                    for s, t in enumerate(OUT[n["cls"]])]
            values = []
            for w in WIDGETS[n["cls"]]:
                if w == "@control":
                    values.append("fixed")
                elif w in n["widgets"]:
                    values.append(n["widgets"][w])
            node = {"id": n["id"], "type": n["cls"], "pos": list(n["pos"]),
                    "size": [460, 600] if n["cls"] == "SnapRefSimple" else [420, 260] if n["cls"] in ("CLIPTextEncode", "Note") else [320, 130],
                    "flags": {}, "order": n["id"] - 1, "mode": 0, "inputs": ins, "outputs": outs,
                    "properties": {"Node name for S&R": n["cls"]}, "widgets_values": values}
            if n["title"]:
                node["title"] = n["title"]
            nodes.append(node)
        links = [[l[0], l[1], l[2], l[3], [i["name"] for i in nodes[l[3] - 1]["inputs"]].index(l[4]), l[5]] for l in links]
        return {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
                "config": {}, "extra": {}, "version": 0.4}

    def api(self) -> dict:
        out = {}
        for n in self.nodes:
            if n["cls"] == "Note":
                continue
            inputs = {k: v for k, v in n["widgets"].items() if k not in ("upload", "snapref_panel")}
            inputs.update({k: [str(src), slot] for k, (src, slot) in n["inputs"].items()})
            out[str(n["id"])] = {"class_type": n["cls"], "inputs": inputs}
        return out


def loaders(g: Graph, x=0):
    unet = g.add("UNETLoader", widgets={"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}, pos=(x, 0))
    clip = g.add("CLIPLoader", widgets={"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}, pos=(x, 160))
    vae = g.add("VAELoader", widgets={"vae_name": "qwen_image_vae.safetensors"}, pos=(x, 320))
    ref = g.add("LoadImage", widgets={"image": "reference.png", "upload": "image"}, pos=(x, 460))
    pos = g.add("CLIPTextEncode", {"clip": (clip, 0)}, {"text": POS}, pos=(x + 360, 0), title="positive (scene only)")
    neg = g.add("CLIPTextEncode", {"clip": (clip, 0)}, {"text": NEG}, pos=(x + 360, 300), title="negative")
    return unet, clip, vae, ref, pos, neg


def ksampler_chain(g, model, pos, neg, vae, latent, prefix, x, y, sampler="euler", steps=28):
    ks = g.add("KSampler", {"model": (model, 0), "positive": (pos, 0), "negative": (neg, 0), "latent_image": (latent, 0)},
               {"seed": SEED, "steps": steps, "cfg": 4.5, "sampler_name": sampler, "scheduler": "simple", "denoise": 1.0}, pos=(x, y))
    dec = g.add("VAEDecode", {"samples": (ks, 0), "vae": (vae, 0)}, pos=(x + 340, y))
    g.add("SaveImage", {"images": (dec, 0)}, {"filename_prefix": prefix}, pos=(x + 560, y))


def simple_node(g, unet, clip, vae, x, y, ref="reference.png", ref2="none", strength_2=1.0, mode="Character & Style"):
    return g.add("SnapRefSimple", {"model": (unet, 0), "clip": (clip, 0), "vae": (vae, 0)},
                 {"adapter": OURS, "reference": ref, "mode": mode, "strength": 1.0, "fidelity": 1.0,
                  "reference_2": ref2, "strength_2": strength_2, "upload": "image", "snapref_panel": ""},
                 pos=(x, y), title="SnapRef")


def ours_node(g, unet, clip, vae, ref, x, y, rg_on=False, extra_refs=()):
    inputs = {"model": (unet, 0), "clip": (clip, 0), "vae": (vae, 0), "image": (ref, 0)}
    inputs.update({f"image_{i + 2}": (r, 0) for i, r in enumerate(extra_refs)})
    return g.add("SnapRef", inputs,
                 {"adapter": OURS, "mode": "cs", "ref_in_uncond": False, "ref_megapixels": 1.0, "strength": 0.0, "ref_caption": "",
                  "ref_guidance_on": rg_on, "ref_guidance": 4.5, "correction": "apg", "ref_end": 1.0,
                  "use_batch": False, "multi_layout": "separate", "weight_1": 1.0, "weight_2": 1.0, "weight_3": 1.0},
                 pos=(x, y), title="SnapRef v1a")


def v2_node(g, unet, vae, ref, x, y):
    return g.add("AnimaRefAdapter", {"model": (unet, 0), "vae": (vae, 0), "image": (ref, 0)},
                 {"lora_name": "anima_ref_adapter_v2_r64.safetensors", "ref_guidance": 4.5, "correction": "apg",
                  "strength": 1.0, "ref_end": 1.0, "ref_megapixels": 1.0}, pos=(x, y), title="ElectronGirl v2")


def rrkv_node(g, unet, vae, pos, neg, ref, x, y, prefix):
    n = g.add("AnimaRoleRefKV", {"model": (unet, 0), "vae": (vae, 0), "positive": (pos, 0), "negative": (neg, 0), "reference": (ref, 0)},
              {"adapter": "anima-role-ref-kv-v1.0.safetensors", "seed": SEED, "width": 832, "height": 1216, "steps": 28, "cfg": 4.5},
              pos=(x, y), title="AnticPan Role-Ref-KV (own sampler)")
    dec = g.add("VAEDecode", {"samples": (n, 0), "vae": (vae, 0)}, pos=(x + 340, y))
    g.add("SaveImage", {"images": (dec, 0)}, {"filename_prefix": prefix}, pos=(x + 560, y))


def build() -> dict[str, Graph]:
    out = {}
    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    lat = g.add("EmptyLatentImage", widgets={"width": 832, "height": 1216, "batch_size": 1}, pos=(360, 600))
    m = ours_node(g, unet, clip, vae, ref, 820, 0)
    ksampler_chain(g, m, pos, neg, vae, lat, "snapref/image", 820, 260)
    g.add("Note", widgets={"text": "SnapRef v1a: one reference image, pose and scene from the prompt.\n\n"
          "- Prompt: quality tags, 1girl/1boy, solo, then the scene. No character name, artist or outfit tags needed.\n"
          "- mode cs: character and style from the reference. char: character only; add artist tags to the prompt.\n"
          "- strength (log scale): +1 to +2 for more reference detail, negative for less.\n"
          "- ref_guidance_on + ref_guidance 6-8: stronger reference at about 1.4x the time; apg keeps colors in check.\n"
          "- ref_caption: optional appearance tags of the reference (no name or artist).\n"
          "- Any sampler works; euler or er_sde, simple, 28-30 steps, cfg 4.5."}, pos=(820, 620))
    out["01_ours_anima_ref_kv"] = g

    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    lat = g.add("EmptyLatentImage", widgets={"width": 832, "height": 1216, "batch_size": 1}, pos=(360, 600))
    m = v2_node(g, unet, vae, ref, 820, 0)
    ksampler_chain(g, m, pos, neg, vae, lat, "electrongirl/v2", 820, 260, sampler="er_sde", steps=30)
    out["02_electrongirl_v2"] = g

    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    ld = g.add("AnimaRefAdapterLoader", {"model": (unet, 0)}, {"lora_name": "anima_ref_adapter_pilot_r32.safetensors",
               "strength": 1.0, "ref_offset_h": 1.0}, pos=(820, 0), title="ElectronGirl pilot_r32")
    rl = g.add("AnimaRefLatent", {"image": (ref, 0), "vae": (vae, 0)}, {"width": 832, "height": 1216, "fill": 255}, pos=(820, 160))
    gd = g.add("AnimaRefGuider", {"model": (ld, 0), "positive": (pos, 0), "negative": (neg, 0), "ref_latent": (rl, 0)},
               {"cfg": 4.5, "ref_guidance": 4.5, "correction": "apg", "rescale_phi": 0.5, "ref_start": 0.0, "ref_end": 1.0,
                "outside": "reference, no extra guidance"}, pos=(1160, 0))
    smp = g.add("KSamplerSelect", widgets={"sampler_name": "euler"}, pos=(1160, 300))
    sch = g.add("BasicScheduler", {"model": (ld, 0)}, {"scheduler": "simple", "steps": 28, "denoise": 1.0}, pos=(1160, 420))
    noise = g.add("RandomNoise", widgets={"noise_seed": SEED}, pos=(1160, 560))
    sca = g.add("SamplerCustomAdvanced", {"noise": (noise, 0), "guider": (gd, 0), "sampler": (smp, 0), "sigmas": (sch, 0),
                "latent_image": (rl, 1)}, pos=(1500, 0))
    dec = g.add("VAEDecode", {"samples": (sca, 0), "vae": (vae, 0)}, pos=(1840, 0))
    g.add("SaveImage", {"images": (dec, 0)}, {"filename_prefix": "electrongirl/pilot"}, pos=(2060, 0))
    out["03_electrongirl_pilot"] = g

    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    rrkv_node(g, unet, vae, pos, neg, ref, 820, 0, "role_ref_kv/v1")
    g.add("Note", widgets={"text": "Role-Ref-KV runs its own Euler loop (shift 3) inside the node: set size, steps, cfg and seed "
          "on the node. It needs the plain Anima base MODEL (no LoRA in front), bf16 or fp32, one reference."}, pos=(820, 320))
    out["04_role_ref_kv"] = g

    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    lat = g.add("EmptyLatentImage", widgets={"width": 832, "height": 1216, "batch_size": 1}, pos=(360, 600))
    m = ours_node(g, unet, clip, vae, ref, 820, 0)
    ksampler_chain(g, m, pos, neg, vae, lat, "compare/snapref", 1160, 0)
    m = v2_node(g, unet, vae, ref, 820, 300)
    ksampler_chain(g, m, pos, neg, vae, lat, "compare/v2", 1160, 300)
    rrkv_node(g, unet, vae, pos, neg, ref, 820, 600, "compare/role_ref_kv")
    ksampler_chain(g, unet, pos, neg, vae, lat, "compare/base", 1160, 900)
    g.add("Note", widgets={"text": "Same reference, prompt, seed and size for every method: SnapRef | ElectronGirl v2 | Role-Ref-KV | "
          "base without reference. Euler/simple 28 steps cfg 4.5 everywhere (Role-Ref-KV: its own Euler loop). "
          "Change the seed on each sampler node together."}, pos=(0, 760))
    out["05_compare_all"] = g

    # two characters: two references side by side in one reference image (no extra training)
    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    ref2 = g.add("LoadImage", widgets={"image": "reference2.png", "upload": "image"}, pos=(0, 760))
    st = g.add("ImageStitch", {"image1": (ref, 0), "image2": (ref2, 0)},
               {"direction": "right", "match_image_size": True, "spacing_width": 0, "spacing_color": "white"}, pos=(360, 600))
    g.nodes[pos - 1]["widgets"]["text"] = ("masterpiece, best quality, score_7, safe, 2girls, the girl on the left is <describe A>, "
                                           "the girl on the right is <describe B>, standing side by side, smile, park, day")
    lat = g.add("EmptyLatentImage", widgets={"width": 1216, "height": 832, "batch_size": 1}, pos=(360, 760))
    m = ours_node(g, unet, clip, vae, st, 820, 0)
    ksampler_chain(g, m, pos, neg, vae, lat, "snapref/two_characters", 820, 300)
    g.add("Note", widgets={"text": "Two characters: load one reference per character (left, right). Image Stitch puts them side by side "
          "into one reference. In the prompt use 2girls / 1boy, 1girl instead of solo and say who is on the left and who "
          "is on the right with a few words each. A landscape size works best."}, pos=(820, 620))
    out["06_ours_two_characters"] = g

    # several references of ONE character (e.g. upper body + full body): encoded separately, K/V concatenated
    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    ref2 = g.add("LoadImage", widgets={"image": "reference_full_body.png", "upload": "image"}, pos=(0, 760))
    lat = g.add("EmptyLatentImage", widgets={"width": 832, "height": 1216, "batch_size": 1}, pos=(360, 600))
    m = ours_node(g, unet, clip, vae, ref, 820, 0, extra_refs=(ref2,))
    ksampler_chain(g, m, pos, neg, vae, lat, "snapref/multi_reference", 820, 300)
    g.add("Note", widgets={"text": "Several references of the SAME character: e.g. an upper-body shot (face, hair) in image and a full-body "
          "shot (shoes, legs) in image_2; image_3 is free for a third. Keep solo in the prompt. multi_layout separate (default) "
          "encodes each at ref_megapixels; stitch puts them side by side in one reference (same speed as one image). "
          "For two DIFFERENT characters use the two_characters workflow instead."}, pos=(820, 620))
    out["07_ours_multi_reference"] = g

    # simple node: the picture(s) are picked inside the node; no LoadImage
    g = Graph(); unet, clip, vae, ref, pos, neg = loaders(g)
    g.nodes[ref - 1]["cls"] = "Note"
    g.nodes[ref - 1]["widgets"] = {"text": "SnapRef: click the picture slot in the SnapRef node (or drop an image on it) to set "
                                           "the character. A second picture of the same character is optional. Write only the scene "
                                           "in the prompt. Mode: Character & Style keeps the picture's art style; Character takes only "
                                           "the character. Strength: how much each picture counts. Fidelity: lower follows the prompt "
                                           "more freely. Every other setting is on SnapRef (Advanced)."}
    lat = g.add("EmptyLatentImage", widgets={"width": 832, "height": 1216, "batch_size": 1}, pos=(360, 600))
    m = simple_node(g, unet, clip, vae, 820, 0)
    ksampler_chain(g, m, pos, neg, vae, lat, "snapref/simple", 1340, 0)
    out["00_snapref_simple"] = g
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--api-dir", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, g in build().items():
        (args.out / f"{name}.json").write_text(json.dumps(g.ui(), ensure_ascii=False, indent=1), encoding="utf-8")
        if args.api_dir:
            args.api_dir.mkdir(parents=True, exist_ok=True)
            (args.api_dir / f"{name}.api.json").write_text(json.dumps(g.api(), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", sorted(build()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

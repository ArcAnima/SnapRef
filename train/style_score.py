"""Style score of style_baseline.py renders with CSD (Somepalli et al. 2024, tomg-group-umd/CSD-ViT-L).

    ~/comfy-venv/bin/python train/style_score.py ~/style_data/style_eval/cases.json ~/ComfyUI/output OUT.json

For each render of artist a:
  own   = mean CSD cosine to the artist's style_targets (same artist, OTHER characters, so identity is not scored)
  other = mean CSD cosine to the style_targets of every other artist
  style = own - other            (> 0: the render looks like the reference artist's style)
Reference ceiling: the same numbers for the reference image itself.
Also reports CLIP-L image cosine to the reference as an identity/content proxy.
CSD = CLIP ViT-L/14 visual backbone with its own copy of the projection ("last_layer_style"), run with transformers'
CLIPVisionModel (same ViT-L/14 layout). The HF checkpoint is a pickle with numpy/argparse objects; it is converted
once to safetensors with an allow-listed weights_only load (CSD_WEIGHTS, default ~/hf/csd_vit_l.safetensors).
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import os

from safetensors.torch import load_file
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModel

PRE = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def openai_to_hf(sd: dict) -> tuple[dict, torch.Tensor]:
    """Map OpenAI-CLIP visual keys (backbone.*) to transformers' CLIPVisionModel keys; return (state, style proj)."""
    out = {}
    g = lambda k: sd["backbone." + k]   # noqa: E731
    out["vision_model.embeddings.class_embedding"] = g("class_embedding")
    out["vision_model.embeddings.patch_embedding.weight"] = g("conv1.weight")
    out["vision_model.embeddings.position_embedding.weight"] = g("positional_embedding")
    out["vision_model.pre_layrnorm.weight"], out["vision_model.pre_layrnorm.bias"] = g("ln_pre.weight"), g("ln_pre.bias")
    out["vision_model.post_layernorm.weight"], out["vision_model.post_layernorm.bias"] = g("ln_post.weight"), g("ln_post.bias")
    n = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith("backbone.transformer.resblocks."))
    for i in range(n):
        p, q = f"transformer.resblocks.{i}.", f"vision_model.encoder.layers.{i}."
        w, b = g(p + "attn.in_proj_weight"), g(p + "attn.in_proj_bias")
        for j, name in enumerate(("q_proj", "k_proj", "v_proj")):
            out[q + f"self_attn.{name}.weight"] = w.chunk(3)[j]
            out[q + f"self_attn.{name}.bias"] = b.chunk(3)[j]
        out[q + "self_attn.out_proj.weight"], out[q + "self_attn.out_proj.bias"] = g(p + "attn.out_proj.weight"), g(p + "attn.out_proj.bias")
        out[q + "layer_norm1.weight"], out[q + "layer_norm1.bias"] = g(p + "ln_1.weight"), g(p + "ln_1.bias")
        out[q + "layer_norm2.weight"], out[q + "layer_norm2.bias"] = g(p + "ln_2.weight"), g(p + "ln_2.bias")
        out[q + "mlp.fc1.weight"], out[q + "mlp.fc1.bias"] = g(p + "mlp.c_fc.weight"), g(p + "mlp.c_fc.bias")
        out[q + "mlp.fc2.weight"], out[q + "mlp.fc2.bias"] = g(p + "mlp.c_proj.weight"), g(p + "mlp.c_proj.bias")
    return out, sd["last_layer_style"]


def csd_scorer(device):
    """Returns emb(PIL image) -> L2-normalised CSD style vector."""
    sd = load_file(os.environ.get("CSD_WEIGHTS", os.path.expanduser("~/hf/csd_vit_l.safetensors")))
    hf_sd, proj = openai_to_hf(sd)
    vit = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    # transformers 4 names keys "vision_model.*", 5 drops the prefix: follow the model's own key names
    own = set(vit.state_dict())
    if not any(k.startswith("vision_model.") for k in own):
        hf_sd = {k[len("vision_model."):]: v for k, v in hf_sd.items()}
    missing, unexpected = vit.load_state_dict(hf_sd, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    vit = vit.to(device).eval().half()
    proj = proj.to(device).half()

    @torch.no_grad()
    def emb(img):
        x = PRE(img.convert("RGB")).unsqueeze(0).to(device).half()
        feat = vit(pixel_values=x).pooler_output   # CLS after post_layernorm, = OpenAI visual before proj
        return F.normalize((feat @ proj).float(), dim=-1)[0]
    return emb


@torch.no_grad()
def main(cases_json: str, render_root: str, out_json: str) -> int:
    dev = "cuda"
    _emb = csd_scorer(dev)
    emb = lambda path: _emb(Image.open(path))  # noqa: E731

    root = Path(cases_json).parent.parent
    d = json.loads(Path(cases_json).read_text(encoding="utf-8"))
    tgt = {c["artist"]: torch.stack([emb(root / p["path"]) for p in c["style_targets"]]) for c in d["cases"]}
    refs = {c["artist"]: emb(root / c["ref"]["path"]) for c in d["cases"]}

    def score(e, a):
        own = (tgt[a] @ e).mean().item()
        other = torch.cat([tgt[b] @ e for b in tgt if b != a]).mean().item()
        return own, other, own - other

    rows = []
    for a in tgt:
        own, other, s = score(refs[a], a)
        rows.append({"artist": a, "method": "reference", "own": own, "other": other, "style": s, "ref_cos": 1.0})
        for f in sorted((Path(render_root) / "style_baseline" / a).glob("*.png")):
            e = emb(f)
            own, other, s = score(e, a)
            rows.append({"artist": a, "method": f.stem.rsplit("_s", 1)[0], "file": f.name, "own": own, "other": other,
                         "style": s, "ref_cos": float(e @ refs[a])})
    summary = {}
    for m in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == m]
        summary[m] = {k: round(sum(r[k] for r in rs) / len(rs), 4) for k in ("own", "other", "style", "ref_cos")}
        summary[m]["n"] = len(rs)
        summary[m]["style_positive_share"] = round(sum(r["style"] > 0 for r in rs) / len(rs), 3)
    Path(out_json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))

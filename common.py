"""Shared pieces for CytoFormer: the UNI2-h image encoder and the per-organ multi-task head.

Design:
  - The ViT-g image encoder (UNI2-h) is finetuned end-to-end during training.
  - The prediction head is PER-ORGAN (multi-task): the organ id is known at input, so each
    organ has its own linear layer over exactly the cell types that occur in it. Output is
    padded to the global classes (out-of-organ classes get -1e4) so loss/eval are uniform.
    The organ id only routes to a head — it does not condition the image features.
"""

import os, json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import timm

_HERE = Path(__file__).resolve().parent

# Backbone (foundation-model) weights — set these env vars to your local paths.
UNI2_WEIGHTS = os.environ.get("CYTOFORMER_UNI2_WEIGHTS", "/path/to/UNI2-h/pytorch_model.bin")

# input normalization (ImageNet statistics, as used by UNI2-h)
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ---- organ / class taxonomy: from the recorded mapping (env-overridable per model) ----
_MAP = json.load(open(os.environ.get("CYTOFORMER_ORGAN_MAP", str(_HERE / "organ_celltype_map.json"))))
ORGANS: List[str] = _MAP["organs"]                     # 16, sorted
GLOBAL_CLASSES: List[str] = _MAP["global_classes"]     # 22, sorted
ORGAN_CELLTYPES = _MAP["organ_to_celltypes"]           # organ -> [celltypes] (sorted, local order)
ORGAN_IDX = {o: i for i, o in enumerate(ORGANS)}
CLASS_IDX = {c: i for i, c in enumerate(GLOBAL_CLASSES)}
NEG = -1e4                                              # logit for a class not in this organ

UNI2_CFG = dict(
    img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
    embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
    mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
    reg_tokens=8, dynamic_img_size=True,
)


def build_uni2(weights_path: Optional[str] = UNI2_WEIGHTS, freeze: bool = False) -> nn.Module:
    """UNI2-h. forward() -> [B,1536] pooled CLS; forward_features() -> [B,265,1536].
    freeze defaults False: the image encoder is finetuned end-to-end."""
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **UNI2_CFG)
    if weights_path and os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"UNI2 load mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    else:                                   # no foundation file -> random init (fine for inference: a full
        print(f"[build_uni2] foundation weights not found ({weights_path}); random init "
              f"(OK if a full checkpoint is loaded on top).", flush=True)
    # grad checkpointing trades compute for memory. With UNI2-h at batch 128 we use only
    # ~13% of a B200's 183GB, so it is wasted recompute (~25-33% slower). Off by default;
    # GRAD_CKPT=1 re-enables it for memory-tight configs.
    if hasattr(model, "set_grad_checkpointing") and os.environ.get("GRAD_CKPT", "0") == "1":
        model.set_grad_checkpointing(True)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


def build_backbone(freeze: bool = False) -> nn.Module:
    """UNI2-h ViT-giant image encoder, forward() -> [B, 1536]."""
    return build_uni2(freeze=freeze)


class PerOrganHead(nn.Module):
    """Multi-task head: one Linear per organ over that organ's cell types. Route by the
    known organ id; scatter each organ head's logits into a [B, n_global] tensor with the
    remaining (out-of-organ) classes set to -1e4 so CE / argmax over the 22 globals is uniform."""

    def __init__(self, d_model: int, dropout: float = 0.25):
        super().__init__()
        self.n_global = len(GLOBAL_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(d_model, len(ORGAN_CELLTYPES[o])) for o in ORGANS])
        for i, o in enumerate(ORGANS):                 # global indices of this organ's classes
            gidx = torch.tensor([CLASS_IDX[c] for c in ORGAN_CELLTYPES[o]], dtype=torch.long)
            self.register_buffer(f"gidx_{i}", gidx, persistent=False)

    def forward(self, feat: torch.Tensor, organ_ids: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]
        out = feat.new_full((B, self.n_global), NEG)
        h = self.drop(feat)
        for oid in torch.unique(organ_ids).tolist():
            mask = organ_ids == oid
            rows = mask.nonzero(as_tuple=True)[0]
            logits = self.heads[oid](h[rows])          # [m, ncls_o]
            gidx = getattr(self, f"gidx_{oid}")
            out[rows.unsqueeze(1), gidx.unsqueeze(0)] = logits.to(out.dtype)
        return out


def summarize(model: nn.Module, name: str) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"name": name, "params_total": total, "params_trainable": train,
            "trainable_frac": train / total}


def smoke(model: nn.Module, name: str, batch: int = 4):
    torch.manual_seed(0)
    x = torch.randn(batch, 3, 224, 224)
    organ_ids = torch.arange(batch) % len(ORGANS)
    model.eval()
    with torch.no_grad():
        logits = model(x, organ_ids)
        feat = model.extract_features(x, organ_ids)
    s = summarize(model, name)
    print(f"[{name}] logits={tuple(logits.shape)} (expect B,{len(GLOBAL_CLASSES)}) "
          f"feat={tuple(feat.shape)} finite={torch.isfinite(logits).all().item()}")
    print(f"[{name}] params total={s['params_total']/1e6:.1f}M "
          f"trainable={s['params_trainable']/1e6:.1f}M ({s['trainable_frac']:.1%})")
    return logits

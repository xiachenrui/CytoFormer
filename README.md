# CytoFormer

Inference code for **CytoFormer**, a cell foundation model that assigns a cell type to every
nucleus on a routine H&E slide. The model was trained on 15.4 million cells from 81 paired
Xenium / H&E sections across 16 organs, using cell types derived from the spatial transcriptome
rather than from manual annotation.

This repository contains the model definition and the inference pipeline. The trained weights are
released separately on Hugging Face.

---

## Model

```
   56 x 56 um H&E crop            ViT-giant encoder                per-organ head
   centred on a nucleus    ->     (UNI2-h init, 1536-d)     ->     (organ selects one)   ->  cell type
        224 x 224 px                                                 16 linear heads
```

| | |
|---|---|
| **Input** | one 56 µm field of view centred on the nucleus, resized to 224×224 px |
| **Encoder** | ViT-giant, patch 14, 1536-d embedding, initialised from UNI2-h and fine-tuned end-to-end |
| **Head** | 16 linear classifiers, one per organ, each over that organ's cell types only (103 outputs in total); the organ identifier selects which head is used and is **not** given to the encoder |
| **Output** | one of 23 global cell types, restricted to the cell types that occur in the given organ |

The organ is a routing signal, not an image feature: the same representation is produced whatever
organ is declared, so the encoder can be reused as a general cell-level feature extractor.

`model.py` defines `CellClassifier`; `common.py` holds the encoder builder, the taxonomy and
`PerOrganHead`.

### Organs and cell types

| organ | cell types |
|---|---|
| bone | Bone_cell, Endothelium, Lymphocyte, Stroma |
| bone_marrow | Endothelium, Erythroid, Myeloid_progenitor, Neutrophil, Stroma |
| brain | Endothelium, Microglia, tumor |
| breast | Endothelium, Epithelium, Lymphocyte, Macrophage, Plasma_cell, Stroma, tumor |
| cervix | Endothelium, Epithelium, Lymphocyte, Macrophage, Plasma_cell, Stroma, tumor |
| colon | Endothelium, Epithelium, Lymphocyte, Macrophage, Plasma_cell, Stroma, tumor |
| heart | Cardiomyocyte, Endothelium, Macrophage, Stroma |
| kidney | Endothelium, Lymphocyte, Macrophage, Plasma_cell, Renal_tubule, Stroma, tumor |
| liver | Cholangiocyte, Endothelium, Hepatocyte, Lymphocyte, Macrophage, Stroma, tumor |
| lung | Chondrocyte, Endothelium, Epithelium, Lymphocyte, Macrophage, Plasma_cell, Stroma, tumor |
| lymph_node | Endothelium, Lymphocyte, Macrophage, Plasma_cell, Squamous_epithelium, Stroma, tumor |
| ovary | Endothelium, Epithelium, Lymphocyte, Macrophage, Plasma_cell, Stroma, tumor |
| pancreas | Acinar, Ductal, Endothelium, Islet, Lymphocyte, Macrophage, Plasma_cell, Stroma, tumor |
| prostate | Endothelium, Epithelium, Lymphocyte, Macrophage, Nerve, Stroma, tumor |
| skin | Endothelium, Epithelium, Lymphocyte, Macrophage, Plasma_cell, Stroma, melanocytic, tumor |
| tonsil | Endothelium, Epithelium, Lymphocyte, Macrophage, Plasma_cell, Stroma |

The full mapping is in `organ_celltype_map.json`.

---

## Install

```bash
git clone https://github.com/zhihuanglab/CytoFormer
cd CytoFormer
pip install -r requirements.txt
```

Download the checkpoint and put it next to `organ_celltype_map.json`:

```bash
mkdir -p checkpoints && cp organ_celltype_map.json checkpoints/
# best.pth (2.6 GB) from https://huggingface.co/zhihuanglab/CytoFormer
huggingface-cli download zhihuanglab/CytoFormer best.pth --local-dir checkpoints
```

The checkpoint contains the whole network, so the UNI2-h foundation weights are **not** needed for
inference.

---

## Usage

### 1. Crop one patch per cell

`crop_cells.py` takes a whole-slide image and a table of nucleus centroids **in that slide's pixel
frame** and writes one 224×224 PNG per cell.

`cells.csv` needs the columns `x`, `y` (and optionally `cell_id`):

```csv
cell_id,x,y
c0,18422,9137
c1,18510,9203
```

```bash
python crop_cells.py \
    --wsi   slide.ome.tif \
    --cells cells.csv \
    --mpp   0.25 \
    --out   patches/
```

`--mpp` is the micrometres per pixel of the slide at level 0; it sets how many pixels the 56 µm
window spans before the patch is resized to 224 px. Use `--fov_um` to change the field of view
(56 µm is what the model was trained with and what works best; see the ablation in the paper).

### 2. Predict

```bash
python infer.py \
    --model_dir checkpoints \
    --patches   patches/ \
    --organ     skin \
    --out       preds.parquet
```

`--organ` must be one of the 16 organs above; predictions are restricted to that organ's cell
types. `--patches` also accepts a list of image files. Add `--batch` to change the batch size
(default 256). The script uses a GPU when one is available and falls back to CPU otherwise.

Output (`preds.parquet`):

| cell_id | file | organ | pred_celltype | prob |
|---|---|---|---|---|
| c0 | c0.png | skin | tumor | 0.9731 |
| c1 | c1.png | skin | Lymphocyte | 0.8442 |

`cell_id` is the image file stem, so predictions join straight back onto your input table.

### From Python

```python
import torch, sys
sys.path.insert(0, ".")
import os; os.environ["CYTOFORMER_ORGAN_MAP"] = "checkpoints/organ_celltype_map.json"
from model import CellClassifier
import common

net = CellClassifier()
sd = torch.load("checkpoints/best.pth", map_location="cpu")["model_state_dict"]
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}   # tolerate a torch.compile prefix
net.load_state_dict(sd); net.eval()

x = torch.randn(2, 3, 224, 224)                                # ImageNet-normalised patches
organ = torch.tensor([common.ORGAN_IDX["skin"]] * 2)
logits = net(x, organ)                                         # (2, 23), out-of-organ classes masked
emb = net.extract_features(x)                                  # (2, 1536) cell embedding
```

`extract_features` returns the cell embedding, which is what we use in the linear-probing and
active-learning experiments in the paper.

---

## Files

| file | what it is |
|---|---|
| `model.py` | `CellClassifier` — encoder + per-organ head |
| `common.py` | encoder builder, taxonomy, `PerOrganHead` |
| `crop_cells.py` | WSI + centroids → 224×224 cell patches |
| `infer.py` | patches → predicted cell types |
| `organ_celltype_map.json` | the 16 organs, 23 global classes and each organ's class list |

## Citation

```bibtex
@article{cytoformer,
  title  = {CytoFormer: A Molecularly Supervised Cell Foundation Model for Histopathology Cell Classification},
  author = {Yao, Jialu and Li, Songhao and Yu, Alina and Huang, Zhi},
  year   = {2026}
}
```

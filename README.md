# PointNet Baseline with Weak Labels on First-Hand LiDAR `.pcd`

A clean, beginner-friendly starter to learn on **your own LiDAR point clouds**.  
This repo clusters raw `.pcd` files with DBSCAN, assigns **weak multi-labels** with simple size rules, converts to `.npy`, and trains a **PointNet baseline** (PyTorch) with reproducible splits. Inference supports **`.pcd` or `.npy`** and **raw or wrapped** checkpoints.

> New to 3D ML? Start here, then iterate. The code is small, readable, and documented.

---
<p align="center">
  <img src="assets/weaklabels.png" alt="Weak label pipeline" width="700"/>
</p>

## ✨ What you get

- **Weak labeling pipeline**: DBSCAN → per-cluster size rules → frame-level multi-labels  
- **PCD → NPY conversion** for fast loading  
- **PointNet baseline** (global max pooling) with `BCEWithLogitsLoss` (multi-label)  
- **Reproducibility**: normalization, batching, deterministic train/val/test split  
- **Inference** on `.pcd` or `.npy`, and works with **raw** or **wrapped** `.pth`  
- **Dev hygiene**: Makefile, minimal CI, `.gitignore`, clear structure  

---

## 📦 Repo structure

```text
.
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ Makefile
├─ .gitignore
├─ .github/workflows/ci.yml
├─ scripts/
│  ├─ cluster_and_label.py      # DBSCAN + size rules → weak labels
│  ├─ convert_pcd_to_npy.py     # .pcd → .npy
│  └─ viz_clusters.py           # optional: colorize clusters
├─ pointnet_baseline/
│  ├─ __init__.py
│  ├─ model.py                  # PointNet baseline (global feature)
│  ├─ data.py                   # Dataset + normalization + splits
│  ├─ utils.py                  # seeding + geometry helpers
│  ├─ train.py                  # training loop + val F1 + best ckpt
│  └─ infer.py                  # inference on .pcd/.npy, raw/wrapped ckpt
├─ data/
│  ├─ raw/                      # put your .pcd here (any subfolders)
│  └─ processed/                # auto-generated .npy
├─ labels/
│  ├─ auto_labels.txt           # generated weak labels (frame → classes)
│  └─ class_map.json            # generated class list & id map
└─ notebooks/
   └─ (optional experiments)
```

> If a folder is empty, add a `.gitkeep` to keep it in Git.

---

## 🚀 Quickstart (copy-paste)

> **Tip for macOS:** Open3D can crash with Python 3.8 in conda. If you see segfaults, use **Python 3.10** (see troubleshooting).

```bash
# 0) From repo root
python -m pip install -r requirements.txt

# 1) Generate weak labels from your .pcd
python scripts/cluster_and_label.py --in data/raw --out labels/auto_labels.txt

# 2) Convert .pcd → .npy
python scripts/convert_pcd_to_npy.py --in data/raw --out data/processed

# 3) Train (deterministic split; tweak epochs/points if needed)
PYTHONPATH=. python -m pointnet_baseline.train \
  --data data/processed --labels labels/auto_labels.txt \
  --epochs 50 --batch_size 8 --points 4096

# 4) Infer (PCD or NPY; works with new best.pth or your own .pth)
PYTHONPATH=. python -m pointnet_baseline.infer \
  --pcd data/processed/your_frame.npy \
  --ckpt runs/best.pth
```

---

## 🧪 How weak labels work (beginner explanation)

1. **DBSCAN** clusters each `.pcd` (noise points get label `-1`).  
2. For each cluster, the code measures the cluster’s **physical size** using a bounding-box diagonal.  
3. That diagonal is compared against `SIZE_RULES` (expected size ranges per class).  
4. Cluster-level candidates are combined into a **frame-level multi-label** set and saved to `labels/auto_labels.txt`.  
5. `labels/class_map.json` is generated so training/inference uses the same class ordering.

> This is a pragmatic way to bootstrap learning **before** you have perfect annotations.

---

## 🔧 Training details

- **Model**: PointNet baseline (per-point MLP → global max pool → MLP head).  
- **Objective**: Multi-label classification (`BCEWithLogitsLoss`).  
- **Normalization**: center to zero-mean; scale to unit sphere.  
- **Augmentations**: light Z-rotation + jitter (train only).  
- **Split**: deterministic (hash-based) into train/val/test.  
- **Metric**: macro-F1 on val; best checkpoint saved as `runs/best.pth`.

---

## 🧠 Inference details

`pointnet_baseline/infer.py` supports:

### Checkpoint formats
- **Wrapped (recommended / produced by train.py)**  
  `{"model": state_dict, "classes": ["a","b",...]}`  
- **Raw (older style)**  
  plain `state_dict` (the script will try to read `labels/class_map.json`; otherwise uses a fallback list)

### Input formats
- **`.pcd`**: reads with Open3D, then normalize & subsample  
- **`.npy`**: bypasses Open3D (useful if Open3D is unstable on your machine)

Examples:
```bash
# new checkpoint produced by train.py
PYTHONPATH=. python -m pointnet_baseline.infer \
  --pcd "data/processed/example.npy" \
  --ckpt "runs/best.pth"

# your older checkpoint (raw state_dict)
PYTHONPATH=. python -m pointnet_baseline.infer \
  --pcd "data/processed/example.npy" \
  --ckpt "runs/pointnet_model.pth"
```

---

## ⚙️ Makefile shortcuts

```bash
make setup      # pip install -r requirements.txt
make label      # DBSCAN weak labels
make convert    # .pcd → .npy
make train      # quick training example
make infer      # example inference on a .npy
make lint       # ruff + black check (non-blocking)
make test       # import smoke test
```

---

## 🛠️ Troubleshooting

### macOS: segfault on `import torch, open3d`
This is common on macOS with Python 3.8 inside conda.  
**Fix**: create a clean env with **Python 3.10**:

```bash
conda create -n pn310 python=3.10 -y
conda activate pn310
pip install --no-cache-dir open3d==0.18.0 torch torchvision numpy tqdm pyyaml
```

Then run the same commands in that env.

### Paths with spaces
Quote them:
```bash
PYTHONPATH=. python -m pointnet_baseline.infer \
  --pcd "data/processed/my file.npy" \
  --ckpt "runs/best.pth"
```

### Module not found
Run modules from repo root and set `PYTHONPATH=.`
```bash
PYTHONPATH=. python -m pointnet_baseline.train ...
```

---

## 🔁 Advanced: changing size rules (vehicles, etc.)

Open `scripts/cluster_and_label.py` and edit `SIZE_RULES`. Example preset for vehicles
(using approximate **3D box diagonal** ranges in meters):

```python
SIZE_RULES = {
    "bicycle":    (1.85, 2.30),
    "motorcycle": (2.35, 2.90),
    "car":        (4.25, 5.60),
    "van":        (5.30, 6.30),
    "truck":      (5.75, 8.05),
    "bus":        (10.7, 12.8),
}
```

> These ranges are intentionally wide for partial/occluded scans. Tune them using your cluster stats.

---

## 🔍 Common CLI flags

```bash
# cluster_and_label.py
--eps 0.2           # DBSCAN radius (m)
--min_pts 30        # DBSCAN min samples

# train.py
--points 4096       # subsampled points per frame
--batch_size 8
--epochs 50
--lr 1e-3
--save_dir runs

# infer.py
--threshold 0.5
--points 4096
```

---

## 📈 Tips for better weak labels

- **Voxel downsample** before DBSCAN (stabilizes clusters).  
- Use a **confidence score** based on distance to size-rule midpoint.  
- Keep top-1 class per cluster if confidence ≥ 0.5.  
- Log `(dx,dy,dz)` and density to refine rules beyond a single diagonal scalar.

---

## 🗺️ Roadmap

- PointNet++ (SA/MSG) upgrade with FPS + ball query  
- Per-cluster JSON output `{class, conf, bbox}`  
- Small sample dataset + visualization notebook  

---

## 🤝 Contributing

Issues and PRs welcome. Good first issues: add size rule presets, per-cluster JSON export, a tiny sample dataset, or unit tests.

---

## 📜 License

MIT (see `LICENSE`).

---

## 🧾 Citation

If you use this repo, please cite the project (see `CITATION.cff`).

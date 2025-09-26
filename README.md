
# PointNet Baseline with Weak Labels on First-Hand PCD

Pipeline:
1) Cluster + size rules → weak multi-labels
2) `.pcd → .npy` conversion
3) Train PointNet baseline with BCEWithLogitsLoss
4) Inference on `.pcd` or `.npy`

## Commands
```bash
# labels
python scripts/cluster_and_label.py --in data/raw --out labels/auto_labels.txt

# convert
python scripts/convert_pcd_to_npy.py --in data/raw --out data/processed

# train
python -m pointnet_baseline.train --data data/processed --labels labels/auto_labels.txt

# infer
python -m pointnet_baseline.infer --pcd data/processed/example.npy --ckpt runs/best.pth
```

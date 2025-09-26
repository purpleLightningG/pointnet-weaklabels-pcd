
import argparse, os, json
import numpy as np, torch
from .model import PointNetBaseline
from .utils import center_scale_unit_sphere

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    # load model & classes (support wrapped or raw state_dict)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        classes = ckpt.get("classes", [])
    else:
        state_dict = ckpt
        classes = []

    if not classes:
        # try to read class map beside ckpt or in labels/
        meta_path = os.path.join(os.path.dirname(args.ckpt), "class_map.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join("labels", "class_map.json")
        if os.path.exists(meta_path):
            classes = json.load(open(meta_path))["classes"]
        else:
            classes = ["chair","table","monitor","pc","box","human"]

    model = PointNetBaseline(num_classes=len(classes))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # accept either .pcd or .npy (lazy-import open3d)
    if args.pcd.lower().endswith(".npy"):
        xyz = np.load(args.pcd).astype(np.float32)
    else:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(args.pcd)
        xyz = np.asarray(pcd.points, dtype=np.float32)

    xyz = center_scale_unit_sphere(xyz)

    n = xyz.shape[0]
    if n >= args.points:
        idx = np.random.choice(n, args.points, replace=False)
    else:
        pad = np.random.choice(n, args.points - n, replace=True)
        idx = np.concatenate([np.arange(n), pad])
    xyz = xyz[idx]

    x = torch.from_numpy(xyz[None, ...]).float()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).numpy()

    preds = [(c,float(p)) for c,p in zip(classes, probs) if p >= args.threshold]
    preds.sort(key=lambda t: t[1], reverse=True)
    print("Predicted labels (>= threshold):")
    for c,p in preds:
        print(f"  {c}: {p:.3f}")

if __name__ == "__main__":
    main()


import argparse, os, glob, json
import numpy as np
from tqdm import tqdm
import open3d as o3d
from sklearn.cluster import DBSCAN

CLASSES = ["chair","table","monitor","pc","box","human"]
SIZE_RULES = {
    "chair":  (0.4, 1.5),
    "table":  (0.7, 2.5),
    "monitor":(0.2, 0.9),
    "pc":     (0.2, 1.0),
    "box":    (0.1, 1.5),
    "human":  (1.2, 2.2),
}

def pcd_paths(root): return glob.glob(os.path.join(root, "**", "*.pcd"), recursive=True)

def cluster_points(xyz, eps=0.2, min_pts=30):
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(xyz)
    return db.labels_

def diag_of_aabb(pts):
    mn = pts.min(0); mx = pts.max(0)
    return float(np.linalg.norm(mx - mn))

def guess_classes_for_cluster(pts):
    d = diag_of_aabb(pts)
    hits = []
    for name,(lo,hi) in SIZE_RULES.items():
        if lo <= d <= hi:
            hits.append((name, abs((lo+hi)/2 - d)))
    hits.sort(key=lambda x:x[1])
    return [h[0] for h in hits[:2]] if hits else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_txt", required=True)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--min_pts", type=int, default=30)
    args = ap.parse_args()

    paths = pcd_paths(args.in_dir)
    assert paths, f"No .pcd found under {args.in_dir}"

    labels_dict = {}
    class_set = set()

    for p in tqdm(paths, desc="labeling"):
        stem = os.path.splitext(os.path.basename(p))[0]
        pcd = o3d.io.read_point_cloud(p)
        if not pcd.has_points():
            continue
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if xyz.shape[0] < args.min_pts:
            continue
        xyz_c = xyz - xyz.mean(0, keepdims=True)
        cluster_ids = cluster_points(xyz_c, eps=args.eps, min_pts=args.min_pts)
        frame_classes = set()
        for cid in np.unique(cluster_ids):
            if cid < 0:  # noise
                continue
            pts = xyz_c[cluster_ids == cid]
            cands = guess_classes_for_cluster(pts)
            frame_classes.update(cands)
        if frame_classes:
            labels_dict[stem] = sorted(frame_classes)
            class_set.update(frame_classes)

    with open(args.out_txt, "w") as f:
        for k in sorted(labels_dict.keys()):
            f.write(f"{k}: {', '.join(labels_dict[k])}\n")

    class_list = sorted(class_set) if class_set else CLASSES
    cmap = {c:i for i,c in enumerate(class_list)}
    with open(os.path.join(os.path.dirname(args.out_txt), "class_map.json"), "w") as f:
        json.dump({"classes": class_list, "map": cmap}, f, indent=2)

    print(f"wrote {len(labels_dict)} labeled frames to {args.out_txt}")
    print(f"classes: {class_list}")

if __name__ == "__main__":
    main()


import argparse, os, glob
import numpy as np
import open3d as o3d
from tqdm import tqdm

def pcd_paths(root):
    return glob.glob(os.path.join(root, "**", "*.pcd"), recursive=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = pcd_paths(args.in_dir)
    assert paths, f"No .pcd found under {args.in_dir}"

    for p in tqdm(paths, desc="convert"):
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.out_dir, stem + ".npy")
        pcd = o3d.io.read_point_cloud(p)
        if not pcd.has_points():
            continue
        xyz = np.asarray(pcd.points, dtype=np.float32)
        np.save(out, xyz)
    print("done.")

if __name__ == "__main__":
    main()

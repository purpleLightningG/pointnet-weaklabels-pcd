
import argparse, os, glob
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def pcd_paths(root):
    return glob.glob(os.path.join(root, "**", "*.pcd"), recursive=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--min_pts", type=int, default=30)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    for p in pcd_paths(args.in_dir):
        pcd = o3d.io.read_point_cloud(p)
        if not pcd.has_points(): 
            continue
        xyz = np.asarray(pcd.points, dtype=np.float32)
        labels = DBSCAN(eps=args.eps, min_samples=args.min_pts).fit_predict(xyz)
        mx = labels.max() + 1
        colors = np.zeros_like(xyz)
        for i in range(mx):
            colors[labels==i] = np.random.RandomState(i).rand(3)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if args.save:
            out = os.path.splitext(p)[0] + "_clustered.pcd"
            o3d.io.write_point_cloud(out, pcd)
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()

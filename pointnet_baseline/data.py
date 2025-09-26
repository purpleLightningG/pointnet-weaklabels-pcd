
import os, glob, json, hashlib
import numpy as np, torch
from torch.utils.data import Dataset
from .utils import center_scale_unit_sphere

def read_labels_txt(path_txt):
    mapping = {}
    with open(path_txt) as f:
        for line in f:
            if ":" not in line: continue
            stem, rhs = line.strip().split(":", 1)
            labels = [t.strip() for t in rhs.split(",") if t.strip()]
            mapping[stem.strip()] = labels
    return mapping

def deterministic_split(keys, ratios=(0.7,0.15,0.15)):
    a,b = int(ratios[0]*100), int(ratios[1]*100)
    train, val, test = [], [], []
    for k in keys:
        h = int(hashlib.sha1(k.encode()).hexdigest(), 16) % 100
        if h < a: train.append(k)
        elif h < a+b: val.append(k)
        else: test.append(k)
    return train, val, test

class FramesDataset(Dataset):
    def __init__(self, npy_dir, labels_txt, class_map_json=None, split="train", points=4096, augment=True):
        self.npy_dir = npy_dir
        self.labels_map = read_labels_txt(labels_txt)
        if class_map_json and os.path.exists(class_map_json):
            meta = json.load(open(class_map_json))
            self.classes = meta["classes"]
        else:
            classes = set()
            for v in self.labels_map.values():
                classes.update(v)
            self.classes = sorted(classes) if classes else []
        self.c2i = {c:i for i,c in enumerate(self.classes)}

        all_npy = glob.glob(os.path.join(npy_dir, "*.npy"))
        stems = [os.path.splitext(os.path.basename(p))[0] for p in all_npy]
        stems = [s for s in stems if s in self.labels_map]

        tr, va, te = deterministic_split(stems)
        pick = {"train":tr, "val":va, "test":te}[split]
        self.stems = sorted(pick)
        self.points = points
        self.augment = augment and (split=="train")

    def __len__(self): return len(self.stems)

    def _subsample(self, xyz):
        n = xyz.shape[0]
        if n >= self.points:
            idx = np.random.choice(n, self.points, replace=False)
        else:
            pad = np.random.choice(n, self.points - n, replace=True)
            idx = np.concatenate([np.arange(n), pad])
        return xyz[idx]

    def __getitem__(self, i):
        stem = self.stems[i]
        path = os.path.join(self.npy_dir, stem + ".npy")
        xyz = np.load(path).astype(np.float32)
        xyz = center_scale_unit_sphere(xyz)
        if self.augment:
            th = np.random.uniform(0, 2*np.pi)
            c,s = np.cos(th), np.sin(th)
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            xyz = xyz @ R.T
            xyz = xyz + np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
        xyz = self._subsample(xyz)

        y = np.zeros(len(self.classes), dtype=np.float32)
        for cls in self.labels_map[stem]:
            if cls in self.c2i:
                y[self.c2i[cls]] = 1.0

        return {
            "points": torch.from_numpy(xyz),
            "target": torch.from_numpy(y),
            "id": stem,
        }

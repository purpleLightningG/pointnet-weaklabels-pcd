
import os, argparse, torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from .data import FramesDataset
from .model import PointNetBaseline
from .utils import set_seed

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/processed")
    ap.add_argument("--labels", required=True, help="labels/auto_labels.txt")
    ap.add_argument("--class_map", default="labels/class_map.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_dir", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def f1_from_logits(logits, targets):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        tp = (preds * targets).sum(0)
        fp = (preds * (1-targets)).sum(0)
        fn = ((1-preds) * targets).sum(0)
        denom = (2*tp + fp + fn + 1e-6)
        f1 = (2*tp) / denom
        return f1.mean().item()

def main():
    args = parse()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    ds_tr = FramesDataset(args.data, args.labels, args.class_map, split="train", points=args.points, augment=True)
    ds_va = FramesDataset(args.data, args.labels, args.class_map, split="val", points=args.points, augment=False)

    num_classes = len(ds_tr.classes)
    assert num_classes > 0, "No classes found. Did you generate labels first?"

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetBaseline(num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "best.pth")

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for batch in tqdm(dl_tr, desc=f"epoch {epoch} train"):
            x = batch["points"].to(device).float()
            y = batch["target"].to(device).float()
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(dl_tr))

        model.eval()
        va_f1 = 0.0; n=0; va_loss=0.0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["points"].to(device).float()
                y = batch["target"].to(device).float()
                logits = model(x)
                va_loss += crit(logits, y).item()
                va_f1 += f1_from_logits(logits, y); n += 1
        va_loss /= max(1, len(dl_va)); va_f1 /= max(1, n)
        print(f"epoch {epoch} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_f1 {va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"model": model.state_dict(), "classes": ds_tr.classes}, best_path)
            print(f"  ✔ new best F1={best_f1:.3f} saved to {best_path}")

    print("training done.")

if __name__ == "__main__":
    main()

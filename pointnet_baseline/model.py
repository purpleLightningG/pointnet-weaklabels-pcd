
import torch, torch.nn as nn

class PointNetBaseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(3,64), nn.ReLU(inplace=True),
            nn.Linear(64,128), nn.ReLU(inplace=True),
            nn.Linear(128,1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024,512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512,256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes),
        )

    def forward(self, x):
        B,P,_ = x.shape
        f = self.feat(x)                  # (B,P,1024)
        f = torch.max(f, dim=1).values    # (B,1024)
        logits = self.head(f)             # (B,C)
        return logits

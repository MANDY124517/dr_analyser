# dr_src/fusion_mlp.py
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from .fusion import collect_logits  # reuses calibrated logits & temps logic

class MLP(nn.Module):
    def __init__(self, in_dim=15, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(tta=True, epochs=100, lr=5e-4, wd=1e-4, bs=64, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    # get calibrated features (optionally with flip TTA)
    X, y, temps = collect_logits(tta=tta)  # X is calibrated stacked logits [N,15]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    dl_val = DataLoader(ds, batch_size=bs, shuffle=False)

    model = MLP(in_dim=X.shape[1], num_classes=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    best = {"f1": 0.0, "qwk": 0.0, "acc": 0.0, "state": None}
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # eval
        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                out = model(xb.to(device)).softmax(dim=1).cpu().numpy().argmax(1)
                preds.append(out); targs.append(yb.numpy())
        pred = np.concatenate(preds); tgt = np.concatenate(targs)
        f1 = f1_score(tgt, pred, average="weighted")
        qwk = cohen_kappa_score(tgt, pred, weights="quadratic")
        acc = accuracy_score(tgt, pred)
        print(f"Epoch {ep}: F1={f1:.3f} | QWK={qwk:.3f} | ACC={acc:.3f}")
        if f1 > best["f1"]:
            best.update({"f1": f1, "qwk": qwk, "acc": acc, "state": model.state_dict()})

    # save best
    out = Path("ckpt") / "fusion_meta_mlp2.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "f1": best["f1"], "qwk": best["qwk"], "acc": best["acc"]}, out)
    print(f"Best MLP fusion: F1={best['f1']:.3f} | QWK={best['qwk']:.3f} | ACC={best['acc']:.3f}")
    print("Saved meta to:", out)

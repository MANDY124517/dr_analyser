# dr_src/fusion.py (upgraded)
from pathlib import Path
import torch, numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from .configs import CFG
from .dataset import AptosDataset
from .transforms import get_val_aug
from .models.rsg_net import RSGRes34DR
from .models.effnet_b3 import EffNetB3DR
from .models.vit_small import ViTSmallDR

def _load_ckpt(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["state_dict"])
    return model.to(device).eval()

@torch.no_grad()
def _get_logits(model, imgs, device):
    with torch.amp.autocast('cuda', enabled=CFG.AMP):
        return model(imgs.to(device)).float()

def _learn_temperature(logits, targets, init_T=1.0):
    # logits: [N, C] torch tensor on CPU; targets: numpy/int
    # optimize single scalar T by grid search (stable, no extra deps)
    import numpy as np
    y = np.asarray(targets)
    best_T, best_nll = init_T, 1e9
    for T in np.linspace(0.5, 2.0, 16):  # coarse grid
        p = (logits / T).softmax(dim=1).clamp_min(1e-9)
        # negative log-likelihood
        nll = -torch.log(p[range(len(y)), torch.as_tensor(y)]).mean().item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    # refine around best
    low, high = max(0.3, best_T - 0.3), min(3.0, best_T + 0.3)
    for T in np.linspace(low, high, 16):
        p = (logits / T).softmax(dim=1).clamp_min(1e-9)
        nll = -torch.log(p[range(len(y)), torch.as_tensor(y)]).mean().item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return float(best_T)

def collect_logits(tta=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(CFG.APTOS_CSV)
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(df, test_size=0.2, stratify=df["diagnosis"], random_state=CFG.SEED)
    val_ds = AptosDataset(va, CFG.APTOS_IMGS, get_val_aug(CFG.IMG_SIZE))
    val_dl = DataLoader(val_ds, batch_size=CFG.VAL_BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY)

    # load base models
    ck = CFG.CKPT_DIR
    rsg = _load_ckpt(RSGRes34DR(CFG.NUM_CLASSES), ck/"rsg_res34"/"best.pt", device)
    eff = _load_ckpt(EffNetB3DR(CFG.NUM_CLASSES), ck/"effnet_b3"/"best.pt", device)
    vit = _load_ckpt(ViTSmallDR(CFG.NUM_CLASSES), ck/"vit_small"/"best.pt", device)

    all_rsg, all_eff, all_vit, all_targs = [], [], [], []

    for imgs, labels in val_dl:
        if tta:
            # original
            l1_r = _get_logits(rsg, imgs, device)
            l1_e = _get_logits(eff, imgs, device)
            l1_v = _get_logits(vit, imgs, device)
            # horizontal flip
            imgs_flipped = torch.flip(imgs, dims=[3])
            l2_r = _get_logits(rsg, imgs_flipped, device)
            l2_e = _get_logits(eff, imgs_flipped, device)
            l2_v = _get_logits(vit, imgs_flipped, device)
            lr = (l1_r + l2_r) / 2.0
            le = (l1_e + l2_e) / 2.0
            lv = (l1_v + l2_v) / 2.0
        else:
            lr = _get_logits(rsg, imgs, device)
            le = _get_logits(eff, imgs, device)
            lv = _get_logits(vit, imgs, device)

        all_rsg.append(lr.cpu()); all_eff.append(le.cpu()); all_vit.append(lv.cpu())
        all_targs.append(labels.cpu())

    Lr = torch.cat(all_rsg, dim=0)
    Le = torch.cat(all_eff, dim=0)
    Lv = torch.cat(all_vit, dim=0)
    y = torch.cat(all_targs, dim=0).numpy()

    # learn temperatures
    Tr = _learn_temperature(Lr.clone(), y, 1.0)
    Te = _learn_temperature(Le.clone(), y, 1.0)
    Tv = _learn_temperature(Lv.clone(), y, 1.0)

    # apply temperatures and convert to numpy for meta
    X = np.concatenate([(Lr/Tr).numpy(), (Le/Te).numpy(), (Lv/Tv).numpy()], axis=1)
    return X, y, (Tr, Te, Tv)

def fit_meta_and_eval(tta=False):
    X, y, temps = collect_logits(tta=tta)
    meta = LogisticRegression(max_iter=1000, C=2.0, multi_class="multinomial", n_jobs=1, solver="lbfgs")
    meta.fit(X, y)
    pred = meta.predict(X)
    f1 = f1_score(y, pred, average="weighted")
    qwk = cohen_kappa_score(y, pred, weights="quadratic")
    acc = accuracy_score(y, pred)
    print(f"Fusion (calibrated, tta={tta}) on val: F1={f1:.3f} | QWK={qwk:.3f} | ACC={acc:.3f}")
    out = Path(CFG.CKPT_DIR) / "fusion_meta_lr_calib2.npz"
    np.savez(out, coef_=meta.coef_, intercept_=meta.intercept_, temps=np.array(temps))
    print("Saved meta to:", out)

import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from tqdm.auto import tqdm
from pathlib import Path
from .configs import CFG
from .dataset import make_loaders
from .losses import FocalLoss
from .utils import init_cuda_safely

def train_one_model(model, model_name):
    device = init_cuda_safely(CFG.SEED)
    train_dl, val_dl = make_loaders()
    model = model.to(device)

    alpha = torch.tensor([0.6, 3.0, 1.2, 6.0, 4.0], device=device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha, num_classes=CFG.NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    if CFG.ONECYCLE:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CFG.LR*10, epochs=CFG.EPOCHS, steps_per_epoch=len(train_dl), pct_start=0.3
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
    scaler = GradScaler(enabled=CFG.AMP)

    best_f1, stalls, patience = -1.0, 0, 6
    ckpt_dir = CFG.CKPT_DIR / model_name; ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(CFG.EPOCHS):
        model.train(); running=0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"{model_name} E{epoch+1}/{CFG.EPOCHS}")
        for step,(imgs,labels) in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(enabled=CFG.AMP):
                logits = model(imgs)
                loss = criterion(logits, labels) / CFG.ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step+1)%CFG.ACCUM_STEPS==0:
                if CFG.GRAD_CLIP:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if CFG.ONECYCLE: scheduler.step()
            running += loss.item()*CFG.ACCUM_STEPS
            if step%40==0: torch.cuda.empty_cache()
            pbar.set_postfix(loss=f"{running/(step+1):.3f}")

        # validation
        model.eval(); preds, targs, vloss= [], [], 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_dl, leave=False, desc="val"):
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(enabled=CFG.AMP):
                    logits = model(imgs); vloss += nn.functional.cross_entropy(logits, labels).item()
                preds.extend(logits.argmax(1).cpu().numpy()); targs.extend(labels.cpu().numpy())
        from numpy import array
        f1 = f1_score(array(targs), array(preds), average="weighted")
        qwk = cohen_kappa_score(array(targs), array(preds), weights="quadratic")
        acc = accuracy_score(array(targs), array(preds))
        print(f"Epoch {epoch+1}: val_loss={vloss/max(1,len(val_dl)):.3f} | F1={f1:.3f} | QWK={qwk:.3f} | ACC={acc:.3f}")

        if f1>best_f1:
            best_f1, stalls = f1, 0
            torch.save({"state_dict":model.state_dict(),"f1":f1,"qwk":qwk}, ckpt_dir/"best.pt")
        else:
            stalls += 1
            if stalls>=patience:
                print("Early stopping for stability."); break
        torch.cuda.empty_cache()
    print(f"{model_name} Best F1: {best_f1:.3f}")

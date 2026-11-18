"""
train.py -- trains ONE run of Setup 1: StyleGAN2-ADA combined training set + DenseNet-121 + CBAM.

Protocol (identical for every setup so the comparison is fair):
  optimizer  Adam, L2 weight decay 1e-4, ONE learning rate for all parameters
  schedule   3 warmup epochs (lr rises linearly from ~0), then cosine decay down to min_lr_ratio x lr
             (lr and min_lr_ratio are the two values tuned by Bayesian Optimization -- manuscript:
              "initial learning rate and decay scheduling")
  batch      16, up to 15 epochs, early stopping on validation AUC with patience 5
  loss       binary cross-entropy weighted by class frequency (pos_weight = n_neg / n_pos; = 1.0 on the
             balanced combined training set)
  epoch      84,384 samples drawn without replacement = one pass of the combined set = 5,274 batches of 16
  model      DenseNet-121 + CBAM + clinical values (densenet_cbam.py)
  no gradient clipping (not in the manuscript or in Chehade et al.)

A run is either
  * a SEARCH TRIAL  (--trial N, called by bo_search.py): validation only, the test set is never opened;
  * a FINAL RUN     (default; called by run_experiments.py): add --eval_test to score the test set once.

Training-set shortcut check: every epoch logs the training recall on REAL vs SYNTHETIC pneumonia. Real recall near 0
with synthetic recall near 1 means the network learned "looks generated" instead of "has pneumonia".

Examples (project root):
    python scripts\\3a-densenet121_stylegan2-ada\\train.py --cbam all --lr 1e-4 --min_lr_ratio 0.1 --seed 42
    python scripts\\3a-densenet121_stylegan2-ada\\train.py --cbam all --params results\\densenet121_stylegan2-ada\\search\\best_params_cbam-all.json --seed 42 --eval_test
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from cxr_data import CXRDataset, COMBINED_CSV, LABELS, ROOT, STATS_JSON, RESULTS_DIR, run_paths
from densenet_cbam import DenseNetVariant

UNTUNED_LR, UNTUNED_FLOOR = 1e-4, 0.0  # used only if neither --params nor --lr is given (printed as a warning)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, c, y, s in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        c = c.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x, c)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def rate(mask, pred):
    return float(pred[mask].mean()) if mask.any() else float("nan")


def build_scheduler(opt, steps_per_epoch, epochs, warmup_epochs, min_lr_ratio):
    """Linear warmup, then cosine decay to min_lr_ratio x lr; stepped once per batch.
    The curve is defined over the full epoch cap, so early stopping simply cuts it short."""
    warmup = max(1, warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, epochs * steps_per_epoch)

    def rule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbam", choices=["none", "last", "all"], default="all")
    ap.add_argument("--lr", type=float, default=None, help="initial learning rate (tuned by Bayesian Optimization)")
    ap.add_argument("--min_lr_ratio", type=float, default=None, help="cosine-decay floor as a fraction of lr (tuned)")
    ap.add_argument("--params", default=None, help="best_params json from bo_search.py (gives lr and min_lr_ratio)")
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15, help="epoch cap")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience (manuscript: 5)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--samples_per_epoch", type=int, default=84384, help="84384 = one pass = 5274 batches of 16")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trial", type=int, default=None, help="set by bo_search.py; makes this a search trial")
    ap.add_argument("--eval_test", action="store_true", help="also score the test set (final runs only)")
    ap.add_argument("--skip_if_done", action="store_true", help="exit if this run's metrics file already exists")
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args()
    assert not (args.trial is not None and args.eval_test), "search trials must never open the test set"

    # ---- hyper-parameters: explicit flag > params file > untuned default
    p = json.load(open(args.params)) if args.params else {}
    lr = args.lr if args.lr is not None else p.get("lr")
    floor = args.min_lr_ratio if args.min_lr_ratio is not None else p.get("min_lr_ratio")
    if lr is None or floor is None:
        lr = UNTUNED_LR if lr is None else lr
        floor = UNTUNED_FLOOR if floor is None else floor
        print(f"WARNING: untuned values in use (lr={lr}, min_lr_ratio={floor}); pass --params or --lr/--min_lr_ratio")

    P = run_paths(args.out, args.cbam, args.seed, args.trial)
    run_id, run_dir = P["run_id"], P["run_dir"]

    rescue = False
    if args.skip_if_done and os.path.exists(P["metrics"]):
        if args.eval_test and not os.path.exists(P["test_pred"]):
            print(f"[{run_id}] metrics exist but no test predictions: evaluating the saved best checkpoint")
            rescue = True
        else:
            print(f"[skip] {run_id} already done")
            return
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    st = json.load(open(STATS_JSON))
    mean, std = st["mean"], st["std"]  # training-set statistics only

    tr = pd.read_csv(COMBINED_CSV)
    va = pd.read_csv(os.path.join(LABELS, "validation_labels.csv"))
    te = pd.read_csv(os.path.join(LABELS, "testing_labels.csv"))
    n_pos = int((tr["Finding Labels"] == "Pneumonia").sum())
    pos_weight = (len(tr) - n_pos) / n_pos  # weighted BCE; equals 1.0 on the balanced set

    g = torch.Generator()
    g.manual_seed(args.seed)
    sampler = RandomSampler(tr, replacement=False, num_samples=args.samples_per_epoch, generator=g)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    tr_dl = DataLoader(CXRDataset(tr, train=True, mean=mean, std=std),
                       batch_size=args.batch, sampler=sampler, drop_last=True, **kw)
    va_dl = DataLoader(CXRDataset(va, os.path.join(ROOT, "validation"), mean=mean, std=std),
                       batch_size=64, shuffle=False, **kw)
    steps_per_epoch = len(tr_dl)

    model = DenseNetVariant(cbam=args.cbam).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)  # one lr for everything
    sched = build_scheduler(opt, steps_per_epoch, args.epochs, args.warmup_epochs, floor)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    print(f"=== {run_id} | lr={lr:.3e} floor={floor:.3f} | batch {args.batch} | {steps_per_epoch} steps/epoch "
          f"| pos_weight {pos_weight:.3f} ===")

    val_aucs, history, best_auc, best_ep, bad, diverged = [], [], -1.0, 0, 0, False
    epochs_to_run = 0 if rescue else args.epochs
    for ep in range(1, epochs_to_run + 1):
        t0 = time.time()
        model.train()
        run_loss, Ps, Ys, Ss = 0.0, [], [], []
        for x, c, y, s in tqdm(tr_dl, desc=f"epoch {ep}/{args.epochs}"):
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            c, y = c.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, c)
            loss = loss_fn(logit.float(), y)
            if not math.isfinite(loss.item()):  # NaN / inf: training blew up (usually lr too high)
                diverged = True
                break
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            Ps.append(torch.sigmoid(logit.float()).detach().cpu().numpy())
            Ys.append(y.cpu().numpy())
            Ss.append(s.numpy())
        if diverged:
            print(f"DIVERGED in epoch {ep}: loss became non-finite")
            break
        Pn, Yn, Sn = np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Ss)
        pred = Pn >= 0.5

        vp, vy = predict(model, va_dl, device)
        val_auc = float(roc_auc_score(vy, vp))
        val_aucs.append(val_auc)
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"], train_loss=run_loss / len(Pn), val_auc=val_auc,
                   # shortcut check: recognising real pneumonia, or only GAN pneumonia?
                   train_recall_real_pn=rate((Yn == 1) & (Sn == 0), pred),
                   train_recall_synth_pn=rate((Yn == 1) & (Sn == 1), pred),
                   train_acc_nf=1 - rate(Yn == 0, pred),
                   epoch_time_sec=time.time() - t0)
        history.append(row)
        pd.DataFrame(history).to_csv(P["history"], index=False)
        print({k: round(float(v), 4) for k, v in row.items()})

        if val_auc > best_auc:
            best_auc, best_ep, bad = val_auc, ep, 0
            torch.save(model.state_dict(), P["best"])
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping at epoch {ep} (best val AUC {best_auc:.4f} at epoch {best_ep})")
                break

    if rescue:
        metrics = json.load(open(P["metrics"]))
    else:
        # search objective: mean of the 3 highest validation AUCs (robust to one lucky epoch; only 31 val positives)
        top3 = float(np.mean(sorted(val_aucs)[-3:])) if val_aucs else 0.5
        metrics = dict(
            run_id=run_id, phase="search" if args.trial is not None else "final", setup="1_densenet121_stylegan2ada",
            cbam=args.cbam, seed=args.seed, trial=args.trial, lr=lr, min_lr_ratio=floor, batch=args.batch,
            warmup_epochs=args.warmup_epochs, epoch_cap=args.epochs, patience=args.patience,
            weight_decay=args.weight_decay, samples_per_epoch=args.samples_per_epoch, pos_weight=pos_weight,
            epochs_run=len(history), best_epoch=best_ep, diverged=diverged,
            best_val_auc=0.5 if diverged else float(best_auc),
            top3_val_auc=0.5 if diverged else top3)  # a diverged run scores chance level (0.5)

    if args.eval_test and not diverged:
        model.load_state_dict(torch.load(P["best"], map_location=device))
        vp, vy = predict(model, va_dl, device)
        fpr, tpr, thr = roc_curve(vy, vp)
        threshold = float(thr[np.argmax(tpr - fpr)])  # Youden's J, chosen on VALIDATION only
        vdf = va.reset_index(drop=True)[["Image Index"]].copy()
        vdf["label"], vdf["prob_pneumonia"] = vy.astype(int), vp
        vdf.to_csv(P["val_pred"], index=False)  # lets evaluate.py pick the ensemble threshold without the test set
        te_dl = DataLoader(CXRDataset(te, os.path.join(ROOT, "testing"), mean=mean, std=std),
                           batch_size=64, shuffle=False, **kw)
        tp, ty = predict(model, te_dl, device)
        tdf = te.reset_index(drop=True)[["Image Index", "Patient ID", "Finding Labels"]].copy()
        tdf["label"], tdf["prob_pneumonia"] = ty.astype(int), tp
        tdf.to_csv(P["test_pred"], index=False)  # evaluate.py: stratified bootstrap, DeLong, ANOVA
        pr = (tp >= threshold).astype(int)
        metrics.update(
            threshold_from_val=threshold, test_auc=float(roc_auc_score(ty, tp)),
            accuracy=float(accuracy_score(ty, pr)), precision=float(precision_score(ty, pr, zero_division=0)),
            recall=float(recall_score(ty, pr)), f1=float(f1_score(ty, pr, zero_division=0)),
            n_test=int(len(ty)), n_test_pneumonia=int(ty.sum()))
    json.dump(metrics, open(P["metrics"], "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""
train.py -- trains ONE run of Setup 4: CycleGAN-cleaned images + ConvNeXt V2-Tiny + CBAM.

Same protocol as Setup 3 (scripts\\3b-convnextv2_stylegan2-ada\\train.py), so the data method is the only thing that
differs. The differences, all forced by the 487-image training set, are written down here and saved in metrics_*.json:
  training set  487 cleaned images (232 pneumonia + 255 No Finding); one epoch = one pass over them
                = 31 batches of 16 (the last batch has 7 images; ConvNeXt has no batch-norm, so that is harmless)
  epochs        cap 50 (Setup 3: 15), warmup 10 epochs (Setup 3: 3 of 15 = 20 %, here 10 of 50 = 20 %), patience 5
  loss          BCE with pos_weight = n_neg / n_pos = 255 / 232 = 1.099 (computed from the table, not typed in)
  normalise     Setup 4's OWN mean/std (make_stats.py on the 487 cleaned training images)
  validation    run B: the 65-image list (31 + 34)      run A: all 5,985 validation images
  test          always all 12,262 images, row order = testing_labels.csv, scored once per final run (--eval_test)
Unchanged: Adam, weight decay 1e-4, one learning rate, batch 16, linear warmup then cosine to min_lr_ratio x lr
(stepped per batch, defined over the epoch cap), bf16, Chehade augmentation on training images only, no gradient
clipping, Youden threshold chosen on the validation set, the model (convnext_cbam.py).

Domain check: every epoch logs training recall / accuracy separately for artifact-domain images (translated by
G_AB) and clean-domain images. If one domain is much worse than the other, the cleaning did not make them equal.

A run is a SEARCH TRIAL (--trial N, called by bo_search.py; never opens the test set) or a FINAL RUN (default).

Examples (project root):
    python scripts\\3c-convnextv2_cyclegan\\train.py --run B --lr 1e-4 --min_lr_ratio 0.1 --seed 42
    python scripts\\3c-convnextv2_cyclegan\\train.py --run B --params results\\convnextv2_cyclegan\\run_B\\search\\best_params_cbam-all.json --seed 42 --eval_test
--smoke = CPU test on synthetic data (no pretrained weights, counts not asserted). Never use it for results.
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

import cxr4_data as D                         # first: it puts the 3b / 2b folders on sys.path
from cxr4_data import CXRDataset, run_paths
from convnext_cbam import ConvNeXtV2Variant   # scripts\3b-... (unchanged Setup 3 model)

C = D.C
UNTUNED_LR, UNTUNED_FLOOR = 1e-4, 0.0  # only if neither --params nor --lr is given (printed as a warning)


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
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(x, c)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def rate(mask, pred):
    return float(pred[mask].mean()) if mask.any() else float("nan")


def build_scheduler(opt, steps_per_epoch, epochs, warmup_epochs, min_lr_ratio):
    """Linear warmup, then cosine decay to min_lr_ratio x lr; stepped once per batch (same as Setup 3).
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
    ap.add_argument("--run", choices=["A", "B"], required=True, help="A = full 5,985 validation, B = 65-image validation")
    ap.add_argument("--cbam", choices=["none", "last", "all"], default="all")
    ap.add_argument("--lr", type=float, default=None, help="initial learning rate (tuned by Bayesian Optimization)")
    ap.add_argument("--min_lr_ratio", type=float, default=None, help="cosine-decay floor as a fraction of lr (tuned)")
    ap.add_argument("--params", default=None, help="best_params json from bo_search.py (gives lr and min_lr_ratio)")
    ap.add_argument("--warmup_epochs", type=int, default=C.WARMUP_EPOCHS)
    ap.add_argument("--epochs", type=int, default=C.EPOCH_CAP, help="epoch cap")
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trial", type=int, default=None, help="set by bo_search.py; makes this a search trial")
    ap.add_argument("--eval_test", action="store_true", help="also score the test set (final runs only)")
    ap.add_argument("--skip_if_done", action="store_true", help="exit if this run's metrics file already exists")
    ap.add_argument("--out", default=None, help="default: results\\convnextv2_cyclegan\\run_<A|B>")
    ap.add_argument("--smoke", action="store_true", help="CPU test on synthetic data only; never for results")
    args = ap.parse_args()
    assert not (args.trial is not None and args.eval_test), "search trials must never open the test set"
    out = args.out or D.run_dir(args.run)

    # ---- hyper-parameters: explicit flag > params file > untuned default
    p = json.load(open(args.params)) if args.params else {}
    lr = args.lr if args.lr is not None else p.get("lr")
    floor = args.min_lr_ratio if args.min_lr_ratio is not None else p.get("min_lr_ratio")
    if lr is None or floor is None:
        lr = UNTUNED_LR if lr is None else lr
        floor = UNTUNED_FLOOR if floor is None else floor
        print(f"WARNING: untuned values in use (lr={lr}, min_lr_ratio={floor}); pass --params or --lr/--min_lr_ratio")

    P = run_paths(out, args.cbam, args.seed, args.trial)
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
    if args.smoke:
        device = torch.device("cpu")
        workers = 0
    else:
        assert torch.cuda.is_available(), "CUDA not available"
        device = torch.device("cuda")
        workers = args.workers
    mean, std = D.load_stats()  # Setup 4's own training-set statistics

    tr, va, te = D.get_tables(args.run, strict=not args.smoke)
    n_pos = int((tr["Finding Labels"] == "Pneumonia").sum())
    pos_weight = (len(tr) - n_pos) / n_pos  # weighted BCE: 255 / 232 = 1.099 on the real data

    g = torch.Generator()
    g.manual_seed(args.seed)
    sampler = RandomSampler(tr, replacement=False, num_samples=len(tr), generator=g)  # one epoch = one pass
    kw = dict(num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    tr_dl = DataLoader(CXRDataset(tr, train=True, mean=mean, std=std),
                       batch_size=args.batch, sampler=sampler, drop_last=False, **kw)
    va_dl = DataLoader(CXRDataset(va, train=False, mean=mean, std=std), batch_size=64, shuffle=False, **kw)
    steps_per_epoch = len(tr_dl)

    model = ConvNeXtV2Variant(cbam=args.cbam, pretrained=not args.smoke).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)  # one lr for everything
    sched = build_scheduler(opt, steps_per_epoch, args.epochs, args.warmup_epochs, floor)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    print(f"=== setup4 run {args.run} | {run_id} | lr={lr:.3e} floor={floor:.3f} | batch {args.batch} | "
          f"{steps_per_epoch} steps/epoch | pos_weight {pos_weight:.3f} | train {len(tr)} val {len(va)} ===")

    val_aucs, history, best_auc, best_ep, bad, diverged = [], [], -1.0, 0, 0, False
    epochs_to_run = 0 if rescue else args.epochs
    for ep in range(1, epochs_to_run + 1):
        t0 = time.time()
        model.train()
        run_loss, n_seen, Ps, Ys, Ss = 0.0, 0, [], [], []
        for x, c, y, s in tqdm(tr_dl, desc=f"epoch {ep}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            c, y = c.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = model(x, c)
            loss = loss_fn(logit.float(), y)
            if not math.isfinite(loss.item()):  # NaN / inf: training blew up (usually lr too high)
                diverged = True
                break
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
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
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"], train_loss=run_loss / max(1, n_seen), val_auc=val_auc,
                   # domain check: is the model equally good on translated (artifact) and untouched (clean) images?
                   train_recall_pn_artifact=rate((Yn == 1) & (Sn == 1), pred),
                   train_recall_pn_clean=rate((Yn == 1) & (Sn == 0), pred),
                   train_acc_nf_artifact=1 - rate((Yn == 0) & (Sn == 1), pred),
                   train_acc_nf_clean=1 - rate((Yn == 0) & (Sn == 0), pred),
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
        # search objective: mean of the 3 highest validation AUCs (robust to one lucky epoch)
        top3 = float(np.mean(sorted(val_aucs)[-3:])) if val_aucs else 0.5
        metrics = dict(
            run_id=run_id, phase="search" if args.trial is not None else "final", setup="4_convnextv2_cyclegan",
            run=args.run, val_set="validation_small_65" if args.run == "B" else "validation_full",
            n_train=int(len(tr)), n_val=int(len(va)), cbam=args.cbam, seed=args.seed, trial=args.trial, lr=lr,
            min_lr_ratio=floor, batch=args.batch, warmup_epochs=args.warmup_epochs, epoch_cap=args.epochs,
            patience=args.patience, weight_decay=args.weight_decay, steps_per_epoch=steps_per_epoch,
            pos_weight=pos_weight, norm_mean=mean, norm_std=std,
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
        te_dl = DataLoader(CXRDataset(te, train=False, mean=mean, std=std), batch_size=64, shuffle=False, **kw)
        tp, ty = predict(model, te_dl, device)
        tdf = te.reset_index(drop=True)[["Image Index", "Patient ID", "Finding Labels"]].copy()
        tdf["label"], tdf["prob_pneumonia"] = ty.astype(int), tp
        tdf.to_csv(P["test_pred"], index=False)  # same columns as Setup 3: evaluate.py pairs on 'Image Index'
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

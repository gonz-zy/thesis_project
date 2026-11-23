"""
SUPPLEMENTARY TEST (Chehade-style random split, copied from 5_chehade_cbam): the full classifier recipe of Chehade et al. (2025, Sec. 3.6), for DenseNet-121 and ConvNeXt V2-Tiny.

  backbone     ImageNet weights; the LOWER layers are frozen (paper: "froze all the weights from the lower convolutional
               layers"; how many is not stated -> assumption: everything before the last block is frozen:
               DenseNet-121 dense blocks 1-3 + transitions, ConvNeXt V2 stem + stages 1-3). Frozen parts run in
               inference mode (BatchNorm statistics fixed, as Keras does for frozen layers).
  CBAM         one CBAM on the last convolution block only (paper Sec. 3.6.2); reduction 16, 7 x 7 spatial conv
  head         global average pooling -> dense 1024 (ReLU) -> dense 512 (ReLU) -> dropout 0.5 -> [+ 3 clinical values]
               -> output. Assumption: the clinical values are joined after the dense layers ("the output of the last
               layer of our DenseNet-121 model"). One logit + sigmoid (paper: 2-unit sigmoid; equivalent for 2 classes).
  training     Adam, fixed lr 1e-4 (no search, no schedule, no weight decay), batch 16, plain binary cross-entropy,
               early stopping patience 5 (epoch cap 50, not stated in the paper), same augmentation as all setups
  input        224 x 224 scaled to [0, 1] only (paper: "normalized in the interval [0; 1]"), no standardization
  evaluation   as in every setup: best-validation epoch, Youden threshold on the split's validation images, its test images

Split / image set: environment variables CXR_SPLIT and CXR_IMAGESET (see ../splitdata.py).
Output:    chehade_recipe/<s2_densenet121|s4_convnextv2>/output/split_<kk>/<image set>/run_B/final/cbam_last/seed<N>/
Run:       CXR_SPLIT=1 CXR_IMAGESET=cyclegan python supplementary_test_for_downsampling/6_chehade_random_split/chehade_recipe/train_chehade.py --net densenet --seed 42
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, RandomSampler

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CH))
for _p in (os.path.join(ROOT, "5_training_and_classification", "s1_densenet121_stylegan2-ada"),
           os.path.join(ROOT, "5_training_and_classification", "s3_convnextv2_stylegan2-ada"),
           os.path.join(ROOT, "3_downsampling_branch"), CH):
    if _p not in sys.path:
        sys.path.append(_p)

import splitdata as D
from splitdata import CXRDataset, run_paths
from densenet_cbam import DenseNetVariant
from convnext_cbam import ConvNeXtV2Variant

NET_DIR = {"densenet": "s2_densenet121", "convnext": "s4_convnextv2"}
LR, BATCH, EPOCH_CAP, PATIENCE = 1e-4, 16, 50, 5


class ChehadeHead(nn.Module):
    def __init__(self, n_feat, n_clinical=3):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(n_feat, 1024), nn.ReLU(inplace=True),
                                   nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.out = nn.Linear(512 + n_clinical, 1)

    def forward(self, f, clin):
        return self.out(torch.cat([self.dense(f), clin], dim=1)).squeeze(1)


class ChehadeDenseNet(DenseNetVariant):
    def __init__(self, pretrained=True):
        super().__init__(cbam="last", pretrained=pretrained)       # CBAM only after dense block 4
        self.head = ChehadeHead(1024)
        self.frozen = nn.ModuleList([self.stem, self.b1, self.t1, self.b2, self.t2, self.b3, self.t3])
        for p in self.frozen.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()                                           # frozen layers stay in inference mode
        return self

    def forward(self, x, clin):
        x = self.stem(x)
        x = self.t1(self.b1(x))
        x = self.t2(self.b2(x))
        x = self.t3(self.b3(x))
        x = self.cbam4(self.b4(x))
        x = F.relu(self.norm5(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1), clin)


class ChehadeConvNeXt(ConvNeXtV2Variant):
    def __init__(self, pretrained=True):
        super().__init__(cbam="last", pretrained=pretrained)       # CBAM only after stage 4
        b = self.backbone
        self.head = ChehadeHead(b.num_features)
        self.frozen = nn.ModuleList([b.stem, b.stages[0], b.stages[1], b.stages[2]])
        for p in self.frozen.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, x, clin):
        b = self.backbone
        x = b.stem(x)
        for stage, cbam in zip(b.stages, self.cbams):
            x = cbam(stage(x))
        x = b.norm_pre(x)
        return self.head(b.forward_head(x, pre_logits=True), clin)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, c, y, s in loader:
        x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(x, c)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def rate(mask, pred):
    return float(pred[mask].mean()) if mask.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=list(NET_DIR), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCH_CAP)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip_if_done", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, NET_DIR[args.net], "output", D.SPLIT_TAG, D.IMAGESET, "run_B")
    P = run_paths(out, "last", args.seed)
    run_id, run_dir = P["run_id"], P["run_dir"]
    if args.skip_if_done and os.path.exists(P["test_pred"]):
        print(f"[skip] {run_id} already done")
        return
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    mean, std = 0.0, 1.0                                             # [0, 1] only, as in the paper
    tr, va, te = D.get_tables("B")
    g = torch.Generator()
    g.manual_seed(args.seed)
    sampler = RandomSampler(tr, replacement=False, num_samples=len(tr), generator=g)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    tr_dl = DataLoader(CXRDataset(tr, train=True, mean=mean, std=std), batch_size=BATCH, sampler=sampler, **kw)
    va_dl = DataLoader(CXRDataset(va, train=False, mean=mean, std=std), batch_size=64, shuffle=False, **kw)

    model = (ChehadeDenseNet() if args.net == "densenet" else ChehadeConvNeXt()).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr, n_all = sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(trainable, lr=LR)                         # fixed lr, no weight decay
    loss_fn = nn.BCEWithLogitsLoss()                                 # plain binary cross-entropy
    print(f"=== Chehade recipe | {args.net} | {D.SPLIT_TAG} | {D.IMAGESET} | {run_id} | lr {LR} fixed | trainable {n_tr:,} of {n_all:,} ===")

    history, best_auc, best_ep, bad = [], -1.0, 0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n_seen, Ps, Ys, Ss = 0.0, 0, [], [], []
        for x, c, y, s in tr_dl:
            x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=True):
                logit = model(x, c)
            loss = loss_fn(logit.float(), y)
            assert math.isfinite(loss.item()), "loss became non-finite"
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
            Ps.append(torch.sigmoid(logit.float()).detach().cpu().numpy())
            Ys.append(y.cpu().numpy())
            Ss.append(s.numpy())
        Pn, Yn, Sn = np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Ss)
        pred = Pn >= 0.5
        vp, vy = predict(model, va_dl, device)
        val_auc = float(roc_auc_score(vy, vp))
        row = dict(epoch=ep, lr=LR, train_loss=run_loss / n_seen, val_auc=val_auc,
                   train_recall_pn_artifact=rate((Yn == 1) & (Sn == 1), pred),
                   train_recall_pn_clean=rate((Yn == 1) & (Sn == 0), pred),
                   train_acc_nf_artifact=1 - rate((Yn == 0) & (Sn == 1), pred),
                   train_acc_nf_clean=1 - rate((Yn == 0) & (Sn == 0), pred), epoch_time_sec=time.time() - t0)
        history.append(row)
        pd.DataFrame(history).to_csv(P["history"], index=False)
        print({k: round(float(v), 4) for k, v in row.items()}, flush=True)
        if val_auc > best_auc:
            best_auc, best_ep, bad = val_auc, ep, 0
            # WEIGHTS: best-validation epoch
            torch.save(model.state_dict(), P["best"])
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {ep} (best val AUC {best_auc:.4f} at epoch {best_ep})")
                break

    model.load_state_dict(torch.load(P["best"], map_location=device))
    vp, vy = predict(model, va_dl, device)
    fpr, tpr, thr = roc_curve(vy, vp)
    threshold = float(thr[np.argmax(tpr - fpr)])                     # Youden's J on validation only
    vdf = va.reset_index(drop=True)[["Image Index"]].copy()
    vdf["label"], vdf["prob_pneumonia"] = vy.astype(int), vp
    vdf.to_csv(P["val_pred"], index=False)
    te_dl = DataLoader(CXRDataset(te, train=False, mean=mean, std=std), batch_size=64, shuffle=False, **kw)
    tp, ty = predict(model, te_dl, device)
    tdf = te.reset_index(drop=True)[["Image Index", "Patient ID", "Finding Labels"]].copy()
    tdf["label"], tdf["prob_pneumonia"] = ty.astype(int), tp
    tdf.to_csv(P["test_pred"], index=False)
    pr = (tp >= threshold).astype(int)
    metrics = dict(run_id=run_id, setup=f"chehade_recipe_{args.net}_{D.SPLIT_TAG}_{D.IMAGESET}", split=D.SPLIT_TAG, recipe="chehade", net=args.net,
                   imageset=D.IMAGESET, cbam="last", seed=args.seed, lr=LR, batch=BATCH, epoch_cap=args.epochs,
                   patience=PATIENCE, trainable_params=n_tr, total_params=n_all, epochs_run=len(history),
                   best_epoch=best_ep, best_val_auc=best_auc, threshold_from_val=threshold,
                   test_auc=float(roc_auc_score(ty, tp)), accuracy=float(accuracy_score(ty, pr)),
                   precision=float(precision_score(ty, pr, zero_division=0)), recall=float(recall_score(ty, pr)),
                   f1=float(f1_score(ty, pr, zero_division=0)), n_test=int(len(ty)))
    json.dump(metrics, open(P["metrics"], "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Kaggle kernel: fine-tune a small cross-encoder on (paper opening, sentence) -> y, the share of the
sentence's word pairs that appear in the authors' abstract. 5 folds matching every CV here (fold column
from prep.py): out-of-fold scores for train, mean of the 5 fold models for test. Runs on TPU (torch_xla)
when the kernel has one, else GPU, else CPU.

Input: /kaggle/input/<dataset>/cands.csv (prep.py). Output: preds.csv (paper_id, split, pos, tf), val_scores.txt.
"""

import glob
import os
import subprocess
import sys
import time

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("TF_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_LEN = int(os.environ.get("TF_MAXLEN", 224))
EPOCHS = int(os.environ.get("TF_EPOCHS", 2))
BS, LR = 32, 3e-5
# TF_FOLDS="2" runs only that fold (Actions CPU: one fold per runner); its test scores are already divided by 5
FOLDS = [int(x) for x in os.environ.get("TF_FOLDS", "0 1 2 3 4").split()]
OUT = os.environ.get("TF_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
LOG = open(os.path.join(OUT, "val_scores.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def device():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
        return dev, xm
    except Exception as e:  # noqa: BLE001
        log("no xla:", repr(e)[:200])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), None


def batches(tok, df, shuffle, rng):
    idx = rng.permutation(len(df)) if shuffle else np.arange(len(df))
    for s in range(0, len(idx), BS):
        b = idx[s:s + BS]
        n = len(b)
        if n < BS:  # fixed shapes keep XLA from recompiling
            if shuffle:
                continue
            b = np.concatenate([b, np.repeat(b[-1:], BS - n)])
        x = tok(df.opening.values[b].tolist(), df.sent.values[b].tolist(), truncation="only_first",
                max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        yield x, torch.tensor(df.y.values[b], dtype=torch.float32), n


def predict(m, tok, df, dev, xm):
    m.eval()
    out = []
    with torch.no_grad():
        for x, _, n in batches(tok, df, False, None):
            p = m(**{k: v.to(dev) for k, v in x.items()}).logits.squeeze(-1)
            out.append(p.float().cpu().numpy()[:n])
    return np.concatenate(out)


def main():
    path = glob.glob("/kaggle/input/**/cands.csv", recursive=True) or [sys.argv[1]]
    c = pd.read_csv(path[0]).fillna("")
    tr, te = c[c.split == "train"].reset_index(drop=True), c[c.split == "test"].reset_index(drop=True)
    if os.environ.get("TF_SMOKE"):  # quick local check: a few papers per fold, short sentences
        keep = set(tr.groupby("fold").paper_id.apply(lambda s: s.drop_duplicates()[:3]).explode())
        tr = tr[tr.paper_id.isin(keep)].reset_index(drop=True)
        te = te[te.paper_id.isin(te.paper_id.drop_duplicates()[:2])].reset_index(drop=True)
    dev, xm = device()
    log("device", dev, "model", MODEL, "max_len", MAX_LEN, "epochs", EPOCHS, "folds", FOLDS, "train rows", len(tr), "test rows", len(te), "cpus", os.cpu_count())
    if xm is None and dev.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained(MODEL)
    tr["tf"] = np.nan
    te_pred = np.zeros(len(te))
    for f in FOLDS:
        t0 = time.time()
        torch.manual_seed(f)
        m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
        fit = tr[tr.fold != f]
        steps = EPOCHS * (len(fit) // BS)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
        lossf = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(f)
        step = 0
        for ep in range(EPOCHS):
            m.train()
            for x, y, _ in batches(tok, fit, True, rng):
                loss = lossf(m(**{k: v.to(dev) for k, v in x.items()}).logits.squeeze(-1), y.to(dev))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                if xm is not None:
                    xm.optimizer_step(opt)
                else:
                    opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 500 == 0:
                    log(f"fold {f} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
        va = tr.fold == f
        tr.loc[va, "tf"] = predict(m, tok, tr[va], dev, xm)
        te_pred += predict(m, tok, te, dev, xm) / 5
        v = tr[va]
        corr = v.groupby("paper_id").apply(lambda g: g.tf.corr(g.y, method="spearman")).mean()
        corr4 = v.groupby("paper_id").apply(lambda g: g.v4.corr(g.y, method="spearman")).mean()
        log(f"fold {f}: within-paper spearman tf {corr:.4f} vs v4 {corr4:.4f}  ({time.time() - t0:.0f}s)")
    te["tf"] = te_pred
    tr = tr[tr.fold.isin(FOLDS)]
    pd.concat([tr, te])[["paper_id", "split", "pos", "tf"]].to_csv(os.path.join(OUT, "preds.csv"), index=False)
    log("done")


if __name__ == "__main__":
    main()

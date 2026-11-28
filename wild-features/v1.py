"""Wild Features v1: v0's table plus per-task means of the acoustic clip features (audio_feats.py).

CV: students are split at random (the data page says students are split between train and test; schools are not
given for test), 5 folds x 3 seeds = 15 paired folds, v0 and v1 on the same folds. The school-grouped score is
printed too (for the write-up question "does it hold across schools").
  python wild-features/v1.py cv <data_dir> <feat_dir>
  python wild-features/v1.py submit <data_dir> <feat_dir> <out.csv>
feat_dir holds train_audio.parquet and test_audio.parquet. Prints aggregates only.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline as B  # noqa: E402

SEEDS = [0, 1, 2]


def audio_table(d, fd, part):
    c = pd.read_csv(f"{d}{part}_clip_metadata.csv")[["filename", "subject_id", "task"]]
    a = pd.read_parquet(f"{fd}{part}_audio.parquet")
    x = c.merge(a, on="filename", how="left")
    keep = os.environ.get("WF_AUDIO")  # comma list of a_* columns to keep (default: all)
    cols = [k for k in a.columns if k.startswith("a_") and (not keep or k in keep.split(","))]
    g = x.groupby(["subject_id", "task"])[cols].mean().unstack("task")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g


def tables(d, fd, part):
    f, s = B.table(d, part)
    return f, f.join(audio_table(d, fd, part)), s


def folds(n, y, g):
    for seed in SEEDS:
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.zeros(n))):
            yield seed, f, a, b


def main():
    mode, d, fd = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3].rstrip("/") + "/"
    f0, f1, s = tables(d, fd, "train")
    y = pd.read_csv(d + "train_labels.csv").set_index("subject_id").outcome_score.reindex(f0.index)
    if mode == "submit":
        t0, t1, _ = tables(d, fd, "test")
        t1 = t1.reindex(columns=f1.columns)
        p = B.fit_predict(f1, y, t1)
        sub = pd.read_csv(d + "submission_format.csv")
        sub["outcome_score"] = pd.Series(p, index=t1.index).reindex(sub.subject_id).to_numpy()
        sub.to_csv(sys.argv[4], index=False)
        print("wrote", len(sub), "rows; features", t1.shape[1])
        return
    print("features v0", f0.shape[1], "v1", f1.shape[1])
    diffs, r0s, r1s = [], [], []
    for seed, f, a, b in folds(len(y), y, None):
        p0 = B.fit_predict(f0.iloc[a], y.iloc[a], f0.iloc[b])
        p1 = B.fit_predict(f1.iloc[a], y.iloc[a], f1.iloc[b])
        r0, r1 = r2_score(y.iloc[b], p0), r2_score(y.iloc[b], p1)
        r0s.append(r0); r1s.append(r1); diffs.append(r1 - r0)
    dd = np.array(diffs)
    t = stats.ttest_1samp(dd, 0, alternative="greater")
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    lo, hi = dd.mean() + np.array([-1, 1]) * stats.t.ppf(0.95, len(dd) - 1) * se
    print(f"v0 {np.mean(r0s):.4f}  v1 {np.mean(r1s):.4f}  diff {dd.mean():+.4f} 90% [{lo:+.4f},{hi:+.4f}] "
          f"p {t.pvalue:.4f} wins {(dd > 0).sum()}/{len(dd)}")
    print("pairs", " ".join(f"{x:+.3f}" for x in dd))
    g = s.school_id.reindex(f0.index)
    for name, ff in (("v0", f0), ("v1", f1)):
        oof = np.zeros(len(y))
        for a, b in GroupKFold(5).split(ff, y, g):
            oof[b] = B.fit_predict(ff.iloc[a], y.iloc[a], ff.iloc[b])
        print(f"school-grouped {name} R2 {r2_score(y, oof):.4f}")


if __name__ == "__main__":
    main()

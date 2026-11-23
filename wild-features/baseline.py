"""Wild Features v0: tabular baseline (no audio) to test the submit path end to end.

Features per student: instrument, the three beginning-of-year task thetas (+ missing flags), and per task the mean
item score, number of clips and mean clip duration. Ridge on standardized features. Every test student is
predicted from its own rows only (no fitting on test).
  python wild-features/baseline.py cv <data_dir>        # 5-fold, grouped by school, R2
  python wild-features/baseline.py submit <data_dir> <out.csv>
Prints aggregates only (the repo and the Actions logs are public).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = ["blending", "deletion", "sentence_repetition"]
THETAS = [f"{t}_theta" for t in TASKS]


def table(d, part):
    s = pd.read_csv(f"{d}{part}_student_metadata.csv")
    c = pd.read_csv(f"{d}{part}_clip_metadata.csv")
    agg = c.groupby(["subject_id", "task"]).agg(mean_score=("item_score", "mean"), n=("item_score", "size"),
                                                 dur=("duration_sec", "mean")).unstack("task")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    x = s.set_index("subject_id").join(agg)
    f = pd.DataFrame(index=x.index)
    f["ktea3"] = (x.instrument == "ktea3").astype(float)
    for col in THETAS + [f"{a}_{t}" for a in ("mean_score", "n", "dur") for t in TASKS]:
        v = x[col] if col in x else pd.Series(np.nan, index=x.index)
        f[col + "_na"] = v.isna().astype(float)
        f[col] = v
    return f, s.set_index("subject_id")


def model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))


def fit_predict(ftr, y, fte):
    med = ftr.median()  # fill with train medians only
    m = model().fit(ftr.fillna(med), y)
    return m.predict(fte.fillna(med))


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    ftr, str_ = table(d, "train")
    y = pd.read_csv(d + "train_labels.csv").set_index("subject_id").outcome_score.reindex(ftr.index)
    if mode == "cv":
        oof = np.zeros(len(y))
        for a, b in GroupKFold(5).split(ftr, y, str_.school_id.reindex(ftr.index)):
            oof[b] = fit_predict(ftr.iloc[a], y.iloc[a], ftr.iloc[b])
        print(f"CV R2 {r2_score(y, oof):.4f}  (n={len(y)}, features {ftr.shape[1]})")
        for k, g in ftr.groupby("ktea3"):
            print(f"  ktea3={int(k)}: n={len(g)}  R2 {r2_score(y[g.index], oof[ftr.index.get_indexer(g.index)]):.4f}")
        return
    fte, _ = table(d, "test")
    sub = pd.read_csv(d + "submission_format.csv")
    p = pd.Series(fit_predict(ftr, y, fte[ftr.columns]), index=fte.index)
    sub["outcome_score"] = p.reindex(sub.subject_id).values
    assert sub.outcome_score.notna().all()
    sub.to_csv(sys.argv[3], index=False)
    print(f"wrote {len(sub)} rows")


if __name__ == "__main__":
    main()

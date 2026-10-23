"""Extractive v8: v5 plus a fine-tuned cross-encoder's score for each candidate sentence.

tf/train.py (Kaggle) fine-tuned a small cross-encoder on (paper opening, sentence) -> y for each paper's
top-60 v4 sentences: out of fold for train (same KFold(5, rs=0) as here), the 5-fold mean for test.
Its score, its rank within the paper and its gap to the paper's best become LightGBM features; sentences
outside the top 60 get NaN.

  python whats-up-docs/v8.py cv     <data_dir>              # 5-fold CV (v5 vs v8)
  python whats-up-docs/v8.py submit <data_dir> <out.csv>
Scores: <data_dir>/../kout-tf-v1/preds.csv (paper_id, split, pos, tf).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import BUDGET, add_lex, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm, summarize  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402

TF = os.environ.get("V8_TF", "kout-tf-v1")


def add_tf(tab, ids, split, d):
    p = pd.read_csv(d + f"../{TF}/preds.csv")
    p = p[p.split == split]
    sc = {(a, int(b)): c for a, b, c in zip(p.paper_id, p.pos, p.tf)}
    out = []
    for i, df in tab.groupby("doc"):
        df = df.copy()
        v = np.array([sc.get((ids[i], k), np.nan) for k in range(len(df))])
        df["tf"] = v
        s = pd.Series(v)
        df["tf_rank"] = (1.0 / s.rank(ascending=False)).values
        df["tf_gap"] = v - np.nanmax(v) if np.isfinite(v).any() else np.nan
        out.append(df)
    return pd.concat(out)


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    tab = add_llm(table(tr.text.tolist(), tr.summary.tolist()), [llm.get(p, "") for p in tr.paper_id])
    tab = add_sel(tab, load_picks(d, tr.paper_id))
    tab = add_tf(tab, list(tr.paper_id), "train", d)
    print("train sentences with a tf score", round(float(tab.tf.notna().mean()), 3), flush=True)
    if mode == "cv":
        v5cols = [c for c in tab.columns if not c.startswith("tf")]
        scores = {}
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
            trn, (te,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
            for name, cols in (("v5", v5cols + ["lex"]), ("v8", list(trn.columns))):
                summ = summarize(fit(trn[cols]), te[cols], len(tr))
                scores.setdefault(name, []).extend(rouge2(tr.summary[i], summ[i]) for i in b)
            print("fold", f, {k: round(np.mean(v), 4) for k, v in scores.items()}, flush=True)
        for k, v in scores.items():
            print(f"{k}: ROUGE-2 F1 {np.mean(v):.4f}")
    else:
        te = pd.read_csv(d + "test_features.csv")
        tt = add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id])
        tt = add_sel(tt, load_picks(d, te.paper_id))
        tt = add_tf(tt, list(te.paper_id), "test", d)
        trn, (tst,) = add_lex(tab, [tt])
        summ = summarize(fit(trn), tst, len(te))
        summ = [s if s.strip() else " ".join(str(t).split()[:BUDGET]) for s, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), "papers with tf", int(tst.groupby("doc").tf.count().gt(0).sum()))


if __name__ == "__main__":
    main()

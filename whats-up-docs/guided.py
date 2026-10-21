"""Extractive v2 with LLM guidance: sentences are re-ranked by how many of their word pairs also
appear in a zero-shot LLM abstract of the same paper (kout-llm-v1, Qwen2.5-7B-Instruct).

  python whats-up-docs/guided.py submit <data_dir> <out.csv>
The LLM output is read from <data_dir>/../kout-llm-v1/preds.csv (Submit Prediction data
"whats-up-docs kout-llm-v1"). Score = within-paper rank of the v2 model + W x rank of the overlap.
On the 100 held-out papers: 0.1801 -> 0.1866 with W = 1.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import BUDGET, add_lex, bigrams, compose, feat_cols, fit, table  # noqa: E402

W = 1.0


def guide_score(df, sc, llm_text):
    g = bigrams(llm_text)
    f = np.array([sum((bigrams(s) & g).values()) / max(sum(bigrams(s).values()), 1) for s in df.sent])
    return pd.Series(sc).rank(pct=True).values + W * pd.Series(f).rank(pct=True).values + 1e-6 * sc


if __name__ == "__main__":
    _, mode, d, out = sys.argv
    d = d.rstrip("/") + "/"
    tr, te = pd.read_csv(d + "train.csv"), pd.read_csv(d + "test_features.csv")
    llm = pd.read_csv(d + "../kout-llm-v1/preds.csv").fillna("").set_index("paper_id").llm
    trn, (tst,) = add_lex(table(tr.text.tolist(), tr.summary.tolist()), [table(te.text.tolist())])
    m = fit(trn)
    summ, n_guided = [""] * len(te), 0
    for i, df in tst.groupby("doc"):
        df = df.reset_index(drop=True)
        sc = m.predict(df[feat_cols(df)])
        pid = te.paper_id.iloc[i]
        if pid in llm.index and llm[pid].strip():
            sc = guide_score(df, sc, llm[pid])
            n_guided += 1
        summ[i] = compose(df, sc, BUDGET)
    summ = [s if s.strip() else " ".join(str(t).split()[:BUDGET]) for s, t in zip(summ, te.text)]
    pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(out, index=False)
    print("wrote", out, len(te), "guided", n_guided)

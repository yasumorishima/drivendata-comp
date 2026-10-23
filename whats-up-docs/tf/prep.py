"""Candidates for the sentence cross-encoder: each paper's top-K sentences by the v4 model
(out of fold for train, with the same KFold(5, rs=0) as every CV here; full fit for test).

  python whats-up-docs/tf/prep.py <data_dir> <out.csv | dataset dir>
Columns: paper_id, split, fold (train only), pos (row in the paper's sentence table), v4, y, opening, sent.
Output stays under ~/drivendata-env (it holds paper text).
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from extractive import add_lex, clean, feat_cols, fit, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402

K = 60
OPEN_WORDS = 80
SENT_WORDS = 90

if __name__ == "__main__":
    d, out = sys.argv[1].rstrip("/") + "/", sys.argv[2]
    if os.path.isdir(out):  # Kaggle Run passes the dataset folder
        out = os.path.join(out, "cands.csv")
    tr = pd.read_csv(d + "train.csv")
    te = pd.read_csv(d + "test_features.csv")
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    tab = add_llm(table(tr.text.tolist(), tr.summary.tolist()), [llm.get(p, "") for p in tr.paper_id])
    tt = add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id])
    rows = []

    def emit(split, fold, t, frame, m):
        for i, df in t.groupby("doc"):
            df = df.reset_index(drop=True)
            sc = pd.Series(m.predict(df[feat_cols(df)]))
            opening = " ".join(str(frame.text.iloc[i]).split()[:OPEN_WORDS])
            for k in sc.nlargest(K).index:
                rows.append({"paper_id": frame.paper_id.iloc[i], "split": split, "fold": fold, "pos": int(k),
                             "v4": float(sc[k]), "y": float(df.y[k]) if "y" in df else 0.0,
                             "opening": opening, "sent": " ".join(clean(df.sent[k]).split()[:SENT_WORDS])})

    trn, (tst,) = add_lex(tab, [tt])
    emit("test", -1, tst, te, fit(trn))
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
        trn, (tb,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
        emit("train", f, tb, tr, fit(trn))
        print("oof fold", f, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("rows", len(rows))

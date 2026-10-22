"""Input for cpu_select.py: for every paper, the top-K candidate sentences by the v4 model
(out of fold for the 1,000 train papers, full fit for the 345 test papers), in document order.

  python whats-up-docs/llm/select_prep.py <data_dir> <out.csv>
Columns: paper_id, split, opening, cands (JSON list of [position in the paper's sentence table, text]).
"""

import json
import os
import sys

import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from extractive import add_lex, clean, feat_cols, fit, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402

K = 30          # candidates per paper
MAX_WORDS = 70  # a candidate longer than this is cut (only in the prompt)

if __name__ == "__main__":
    d, out = sys.argv[1].rstrip("/") + "/", sys.argv[2]
    tr = pd.read_csv(d + "train.csv")
    te = pd.read_csv(d + "test_features.csv")
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    tab = add_llm(table(tr.text.tolist(), tr.summary.tolist()), [llm.get(p, "") for p in tr.paper_id])
    tt = add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id])
    rows = []

    def emit(split, t, frame, m):
        for i, df in t.groupby("doc"):
            df = df.reset_index(drop=True)
            sc = m.predict(df[feat_cols(df)])
            top = sorted(pd.Series(sc).nlargest(K).index)
            cands = [[int(k), " ".join(clean(df.sent[k]).split()[:MAX_WORDS])] for k in top]
            rows.append({"paper_id": frame.paper_id.iloc[i], "split": split,
                         "opening": " ".join(str(frame.text.iloc[i]).split()[:120]),
                         "cands": json.dumps(cands, ensure_ascii=False)})

    trn, (tst,) = add_lex(tab, [tt])
    emit("test", tst, te, fit(trn))
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
        trn, (tb,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
        emit("train", tb, tr, fit(trn))
        print("oof fold", f, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("rows", len(rows))

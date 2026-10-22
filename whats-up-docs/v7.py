"""Extractive v7: v5 plus meaning-level match of each sentence (emb.py: MiniLM cosine to the LLM
abstract, the opening and the paper's mean sentence, and their within-paper ranks).

  python whats-up-docs/v7.py cv     <data_dir>
  python whats-up-docs/v7.py submit <data_dir> <out.csv>
Embedding features: <data_dir>/../kout-emb-v1/preds.csv (paper_id, pos, e_llm, e_open, e_cent).
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

EMB = os.environ.get("V7_EMB", "kout-emb-v1")
ECOLS = ["e_llm", "e_open", "e_cent"]


def add_emb(tab, ids, emb):
    t = tab.copy()
    t["paper_id"] = [ids[i] for i in t.doc]
    t["pos"] = t.groupby("doc").cumcount()
    t = t.reset_index().merge(emb, on=["paper_id", "pos"], how="left").set_index("index")
    for c in ECOLS:
        t[c + "_r"] = t.groupby("doc")[c].rank(pct=True)
    return t.drop(columns=["paper_id", "pos"])


def build(d, frame, abstracts=None):
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    emb = pd.read_csv(d + f"../{EMB}/preds.csv")
    t = add_llm(table(frame.text.tolist(), abstracts), [llm.get(p, "") for p in frame.paper_id])
    t = add_sel(t, load_picks(d, frame.paper_id))
    return add_emb(t, list(frame.paper_id), emb)


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    tab = build(d, tr, tr.summary.tolist())
    if mode == "cv":
        v5cols = [c for c in tab.columns if not c.startswith("e_")]
        scores = {}
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
            trn, (te,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
            for name, cols in (("v5", v5cols + ["lex"]), ("v7", list(trn.columns))):
                summ = summarize(fit(trn[cols]), te[cols], len(tr))
                scores.setdefault(name, []).extend(rouge2(tr.summary[i], summ[i]) for i in b)
            print("fold", f, {k: round(np.mean(v), 4) for k, v in scores.items()}, flush=True)
        for k, v in scores.items():
            print(f"{k}: ROUGE-2 F1 {np.mean(v):.4f}")
    else:
        te = pd.read_csv(d + "test_features.csv")
        trn, (tst,) = add_lex(tab, [build(d, te)])
        summ = summarize(fit(trn), tst, len(te))
        summ = [s if s.strip() else " ".join(str(t).split()[:BUDGET]) for s, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te))


if __name__ == "__main__":
    main()

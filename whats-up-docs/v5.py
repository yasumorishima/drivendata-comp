"""Extractive v5: v4 plus which sentences a zero-shot LLM picked for the abstract.

cpu_select.py showed Qwen2.5-7B (4-bit) the paper's top-30 v4 sentences and asked for the 6-10 the
authors' abstract would contain (the 2nd-place public solution does this with a paid LLM). Whether a
sentence was picked, and how early, become LightGBM features next to v4's LLM-abstract overlap.

  python whats-up-docs/v5.py cv     <data_dir>              # 5-fold CV (v4 vs v5)
  python whats-up-docs/v5.py submit <data_dir> <out.csv>
Picks: <data_dir>/../kout-llm-sel-v1/preds.csv (paper_id, split, llm = sentence positions, best first).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import BUDGET, add_lex, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm, summarize  # noqa: E402

SEL = os.environ.get("V5_SEL", "kout-llm-sel-v1")


def add_sel(tab, picks):
    """picks[doc] = list of sentence positions within the paper, most important first."""
    out = []
    for i, df in tab.groupby("doc"):
        df = df.copy()
        order = {p: k for k, p in enumerate(picks[i])}
        pos = np.arange(len(df))
        df["sel"] = [float(p in order) for p in pos]
        df["sel_rank"] = [1.0 / (1 + order[p]) if p in order else 0.0 for p in pos]
        df["sel_n"] = len(order)
        out.append(df)
    return pd.concat(out)


def load_picks(d, ids):
    s = pd.read_csv(d + f"../{SEL}/preds.csv").fillna("")
    s = s.set_index("paper_id").llm.astype(str)
    return [[int(x) for x in s.get(p, "").split() if x.isdigit()] for p in ids]


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    tab = add_llm(table(tr.text.tolist(), tr.summary.tolist()), [llm.get(p, "") for p in tr.paper_id])
    tab = add_sel(tab, load_picks(d, tr.paper_id))
    if mode == "cv":
        v4cols = [c for c in tab.columns if not c.startswith("sel")]
        scores = {}
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
            trn, (te,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
            for name, cols in (("v4", v4cols + ["lex"]), ("v5", list(trn.columns))):
                summ = summarize(fit(trn[cols]), te[cols], len(tr))
                scores.setdefault(name, []).extend(rouge2(tr.summary[i], summ[i]) for i in b)
            print("fold", f, {k: round(np.mean(v), 4) for k, v in scores.items()}, flush=True)
        for k, v in scores.items():
            print(f"{k}: ROUGE-2 F1 {np.mean(v):.4f}")
    else:
        te = pd.read_csv(d + "test_features.csv")
        tt = add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id])
        tt = add_sel(tt, load_picks(d, te.paper_id))
        trn, (tst,) = add_lex(tab, [tt])
        summ = summarize(fit(trn), tst, len(te))
        summ = [s if s.strip() else " ".join(str(t).split()[:BUDGET]) for s, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), "papers with picks", int((tst.groupby("doc").sel_n.first() > 0).sum()))


if __name__ == "__main__":
    main()

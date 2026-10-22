"""Extractive v6: v5 plus a second zero-shot LLM abstract written with a different prompt
(cpu_abstract2.py: built by copying the wording of the top-30 v4 sentences).

  python whats-up-docs/v6.py cv     <data_dir>              # 5-fold CV (v5 vs v6)
  python whats-up-docs/v6.py submit <data_dir> <out.csv>
Second abstract: <data_dir>/../kout-llm-abs2-v1/preds.csv (paper_id, split, llm).
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

ABS2 = os.environ.get("V6_ABS2", "kout-llm-abs2-v1")


def add_llm2(tab, texts):
    """Same overlap features as v4, against the second abstract (columns llm2_*), plus the max of both."""
    base = set(tab.columns)
    t = add_llm(tab.drop(columns=[c for c in tab.columns if c.startswith("llm_")]), texts)
    t = t.rename(columns={c: "llm2_" + c[4:] for c in t.columns if c.startswith("llm_")})
    for c in [c for c in base if c.startswith("llm_")]:
        t[c] = tab.loc[t.index, c]
    t["llmx_bg"] = np.maximum(t.llm_bg, t.llm2_bg)
    return t


def build(d, frame, abstracts=None):
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    ab2 = pd.read_csv(d + f"../{ABS2}/preds.csv").fillna("").set_index("paper_id").llm
    t = table(frame.text.tolist(), abstracts)
    t = add_llm(t, [llm.get(p, "") for p in frame.paper_id])
    t = add_llm2(t, [ab2.get(p, "") for p in frame.paper_id])
    return add_sel(t, load_picks(d, frame.paper_id))


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    tab = build(d, tr, tr.summary.tolist())
    if mode == "cv":
        v5cols = [c for c in tab.columns if not c.startswith(("llm2_", "llmx_"))]
        scores = {}
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
            trn, (te,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
            for name, cols in (("v5", v5cols + ["lex"]), ("v6", list(trn.columns))):
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

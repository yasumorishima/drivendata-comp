"""Extractive v4: the v2 LightGBM with features from a zero-shot LLM abstract of the same paper.

The LLM (Qwen2.5-7B-Instruct, never trained on these abstracts) says what the paper is about; a
sentence that shares many word pairs with that text is likely to be close to the authors' abstract.
v3 (guided.py) mixed this in by hand (LB 0.1822); here LightGBM learns how to use it.

  python whats-up-docs/v4.py cv     <data_dir>              # 5-fold CV of ROUGE-2 F1 (v2 vs v4)
  python whats-up-docs/v4.py submit <data_dir> <out.csv>
LLM text: <data_dir>/../kout-llm-cpu-v1/preds.csv (paper_id, split, llm) for all train and test papers.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import BUDGET, add_lex, bigrams, compose, fit, rouge2, table, tokens  # noqa: E402

LLM = os.environ.get("V4_LLM", "kout-llm-cpu-v1")


def add_llm(tab, texts):
    """texts[doc] = LLM abstract; adds overlap features (raw and within-paper rank)."""
    out = []
    for i, df in tab.groupby("doc"):
        df = df.copy()
        g, u = bigrams(texts[i]), set(tokens(texts[i]))
        bs = [bigrams(s) for s in df.sent]
        us = [set(tokens(s)) for s in df.sent]
        df["llm_bg"] = [sum((b & g).values()) / max(sum(b.values()), 1) for b in bs]
        df["llm_bgc"] = [sum((b & g).values()) for b in bs]
        df["llm_uni"] = [len(x & u) / max(len(x), 1) for x in us]
        df["llm_bg_r"] = df.llm_bg.rank(pct=True)
        df["llm_uni_r"] = df.llm_uni.rank(pct=True)
        df["llm_len"] = len(tokens(texts[i]))
        out.append(df)
    return pd.concat(out)


def summarize(m, tab, n_docs):
    from extractive import feat_cols
    res = [""] * n_docs
    for i, df in tab.groupby("doc"):
        df = df.reset_index(drop=True)
        res[i] = compose(df, m.predict(df[feat_cols(df)]), BUDGET)
    return res


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    tab = table(tr.text.tolist(), tr.summary.tolist())
    tab = add_llm(tab, [llm.get(p, "") for p in tr.paper_id])
    if mode == "cv":
        base = [c for c in tab.columns if not c.startswith("llm_")]
        scores = {"v2": [], "v4": []}
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
            trn, (te,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
            for name, cols in (("v2", base + ["lex"]), ("v4", list(trn.columns))):
                m = fit(trn[cols])
                summ = summarize(m, te[cols], len(tr))
                scores[name] += [rouge2(tr.summary[i], summ[i]) for i in b]
            print("fold", f, {k: round(np.mean(v), 4) for k, v in scores.items()}, flush=True)
        for k, v in scores.items():
            print(f"{k}: ROUGE-2 F1 {np.mean(v):.4f}")
    else:
        te = pd.read_csv(d + "test_features.csv")
        tt = add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id])
        trn, (tst,) = add_lex(tab, [tt])
        m = fit(trn)
        summ = summarize(m, tst, len(te))
        summ = [s if s.strip() else " ".join(str(t).split()[:BUDGET]) for s, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), "with llm text", sum(bool(llm.get(p, "")) for p in te.paper_id))


if __name__ == "__main__":
    main()

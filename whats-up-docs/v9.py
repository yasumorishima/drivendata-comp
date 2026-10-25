"""Extractive v9: v5's sentence scores, but the summary is chosen as a set instead of sentence by sentence.

v4/v5 take sentences in score order until 190 words (skipping near duplicates). That ignores what the
other chosen sentences already cover. Here the summary is built by budgeted greedy on a coverage objective
(the usual submodular recipe for extractive summaries, Lin & Bilmes 2010):
  value(S) = sum over distinct word pairs b in S of max score of a sentence in S containing b   (expected hits)
           + lam * number of distinct word pairs of the zero-shot LLM abstract that S covers
and each step adds the candidate with the best gain / words**r, among the top N by score, until 190 words.

  python whats-up-docs/v9.py cv     <data_dir>              # 5-fold CV: v5 compose vs set selection
  python whats-up-docs/v9.py submit <data_dir> <out.csv>    # env V9_LAM, V9_R, V9_N (default 0, 1, 40)
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import BUDGET, add_lex, bigrams, clean, compose, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402


def select(df, scores, llm_text, lam, r, n_top, budget=BUDGET):
    order = np.argsort(-scores)[:n_top]
    g = set(bigrams(llm_text))
    cand = {k: (set(bigrams(df.sent.iloc[k])), max(int(df.words.iloc[k]), 1)) for k in order}
    best = {}          # word pair -> best score among chosen sentences containing it
    covered = set()    # LLM-abstract word pairs already covered
    chosen, words = [], 0
    while cand and words < budget:
        def gain(k):
            b, w = cand[k]
            s = max(scores[k], 0.0)
            hit = sum(s - best.get(x, 0.0) for x in b if s > best.get(x, 0.0))
            return (hit + lam * len((b & g) - covered)) / w ** r
        k = max(cand, key=gain)
        if gain(k) <= 0:
            break
        b, w = cand.pop(k)
        for x in b:
            best[x] = max(best.get(x, 0.0), max(scores[k], 0.0))
        covered |= b & g
        chosen.append(k)
        words += w
    return " ".join(clean(x) for x in df.sent.iloc[sorted(chosen)])


def tables(d):
    tr = pd.read_csv(d + "train.csv")
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    texts = [llm.get(p, "") for p in tr.paper_id]
    tab = add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts)
    return tr, llm, texts, add_sel(tab, load_picks(d, tr.paper_id))


def oof(d):
    """Out-of-fold v5 sentence scores for the 1,000 train papers (cached outside the repo)."""
    cache = os.path.expanduser("~/drivendata-env/data/v9-oof.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    tr, llm, texts, tab = tables(d)
    res = {}
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
        trn, (te,) = add_lex(tab[tab.doc.isin(a)], [tab[tab.doc.isin(b)]])
        m = fit(trn)
        for i, df in te.groupby("doc"):
            df = df.reset_index(drop=True)
            res[i] = (df[["sent", "words"]], m.predict(df[feat_cols(df)]), f)
        print("oof fold", f, flush=True)
    out = (list(tr.summary), texts, res)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    if mode == "cv":
        refs, texts, res = oof(d)
        configs = [("v5 compose", None)] + [(f"set lam={lam} r={r} n={n}", (lam, r, n))
                                             for n in (20, 40) for r in (0.0, 1.0) for lam in (0.0, 0.3, 1.0)]
        for name, c in configs:
            sc, per_fold = [], [[] for _ in range(5)]
            for i, (df, s, f) in res.items():
                summ = compose(df, s) if c is None else select(df, s, texts[i], *c)
                v = rouge2(refs[i], summ)
                sc.append(v)
                per_fold[f].append(v)
            print(f"{name:28s} {np.mean(sc):.4f}  folds " + " ".join(f"{np.mean(p):.4f}" for p in per_fold), flush=True)
    else:
        # default = the CV winner (10-24): no LLM-coverage term, gain per word, top 40 candidates
        lam, r, n = (float(os.environ.get("V9_LAM", 0)), float(os.environ.get("V9_R", 1)), int(os.environ.get("V9_N", 40)))
        tr, llm, texts, tab = tables(d)
        te = pd.read_csv(d + "test_features.csv")
        tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), load_picks(d, te.paper_id))
        trn, (tst,) = add_lex(tab, [tt])
        m = fit(trn)
        summ = [""] * len(te)
        for i, df in tst.groupby("doc"):
            df = df.reset_index(drop=True)
            summ[i] = select(df, m.predict(df[feat_cols(df)]), llm.get(te.paper_id[i], ""), lam, r, n)
        summ = [s if s.strip() else " ".join(str(t).split()[:BUDGET]) for s, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"lam={lam} r={r} n={n}")


if __name__ == "__main__":
    main()

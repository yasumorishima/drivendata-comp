"""Extractive v26: re-measure v19's evidence (LLM-picked sentences sel_*, agreement of LLM abstracts agree*) on
top of v21, with the comparison rule of 2026-11-03: 5 folds x 3 bag seeds (15 paired differences).

v19 on v18 was +0.0006 on one 5-fold run (p 0.23), held as "maybe". Its agreement columns are paper-level; with
four abstracts the agreement is taken over all of them (mean pairwise Jaccard of abstracts 1-4, and each with the
opening).

  PYTHONHASHSEED=0 python whats-up-docs/v26.py cv <data_dir>        # prints per seed x fold, v21 vs arms
  python whats-up-docs/v26.py submit <data_dir> <out.csv>             # env V26_SET (default both)

11-14, 15 pairs vs v21: sel +0.0007 (p 0.065), agree +0.0003 (p 0.16), both +0.0015 (13/15 wins, p 0.0001).
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
import v15  # noqa: E402
import v17  # noqa: E402
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402
from v19 import AGR, SEL, add19, jac  # noqa: E402
from v21 import ABS4, SETS as S21, add34, load3  # noqa: E402

AGR4 = ["agree_all", "agree_open"]
BASE = S21["v21"]
SETS = {"v21": BASE, "sel": BASE + SEL, "agree": BASE + AGR + AGR4, "both": BASE + SEL + AGR + AGR4}
SEEDS = [int(x) for x in os.environ.get("V26_SEEDS", "0,10,20").split(",")]


def feats(df, text, s, l1, opening, picks, t2, t3, t4, ref=None, cached=True):
    full = v14.full(df, text) if cached else df  # the cached train table lacks position columns (v21 does the same)
    r, sents, _ = v15.rows(full, text, s, l1, opening, ref)
    r = add34(v17.add2(r, sents, t2), sents, t3, t4)
    r = add19(r, full, picks, l1, t2, opening)
    abs_ = [t for t in (l1, t2, t3, t4) if t]
    js = [jac(a, b) for a, b in combinations(abs_, 2)]
    r["agree_all"] = np.nanmean(js) if js else np.nan
    r["agree_open"] = np.nanmean([jac(a, opening) for a in abs_]) if abs_ else np.nan
    return r, sents


def submit(d, tr, texts, built, pk, a2, a3, a4, out):
    fs = SETS[os.environ.get("V26_SET", "both")]
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), pk)
    te = pd.read_csv(d + "test_features.csv")
    tpk = load_picks(d, te.paper_id)
    tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), tpk)
    trn, (tst,) = add_lex(tab, [tt])
    sm = fit(trn)
    allrows = [built[i][0] for i in built]
    m = Bag(fs).fit(v10.with_te(allrows))
    summ = [""] * len(te)
    for i, df in tst.groupby("doc"):
        df = df.reset_index(drop=True)
        p = te.paper_id[i]
        r, sents = feats(df, te.text[i], sm.predict(df[feat_cols(df)]), llm.get(p, ""),
                         " ".join(str(te.text[i]).split()[:300]), tpk[i], a2.get(p, ""), a3.get(p, ""), a4.get(p, ""), cached=False)
        v10.target_encode(allrows, [r])
        summ[i] = choose_units(r, sents, mix(r, m.predict(r), W))
    summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
    pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(out, index=False)
    print("wrote", out, len(te), "set", fs[-3:])


def main():
    d = sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4 = v17.load2(d), load3(d), load3(d, ABS4)
    pk = load_picks(d, tr.paper_id)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {}
    for i, (df, s, f) in res.items():
        p = tr.paper_id[i]
        built[i] = feats(df, tr.text[i], s, texts[i], openings[i], pk[i], a2.get(p, ""), a3.get(p, ""), a4.get(p, ""),
                         refs[i])
    print("built", len(built), flush=True)
    if sys.argv[1] == "submit":
        return submit(d, tr, texts, built, pk, a2, a3, a4, sys.argv[3])
    for seed in SEEDS:
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([built[i][0] for i in tr_ids])
            ms = {k: Bag(fs, seed).fit(trows) for k, fs in SETS.items()}
            erows = [built[i][0].copy() for i in te_ids]
            v10.target_encode([built[i][0] for i in tr_ids], erows)
            sc = {k: [] for k in SETS}
            for i, r in zip(te_ids, erows):
                for k, m in ms.items():
                    sc[k].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, m.predict(r), W))))
            print(f"seed {seed} fold {f} " + " ".join(f"{k} {np.mean(v):.5f}" for k, v in sc.items()), flush=True)


if __name__ == "__main__":
    main()

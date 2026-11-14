"""Extractive v26: re-measure v19's evidence (LLM-picked sentences sel_*, agreement of LLM abstracts agree*) on
top of v21, with the comparison rule of 2026-11-03: 5 folds x 3 bag seeds (15 paired differences).

v19 on v18 was +0.0006 on one 5-fold run (p 0.23), held as "maybe". Its agreement columns are paper-level; with
four abstracts the agreement is taken over all of them (mean pairwise Jaccard of abstracts 1-4, and each with the
opening).

  PYTHONHASHSEED=0 python whats-up-docs/v26.py cv <data_dir>        # prints per seed x fold, v21 vs arms
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
from extractive import rouge2  # noqa: E402
from v5 import load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402
from v19 import AGR, SEL, add19, jac  # noqa: E402
from v21 import ABS4, SETS as S21, add34, load3  # noqa: E402

AGR4 = ["agree_all", "agree_open"]
BASE = S21["v21"]
SETS = {"v21": BASE, "sel": BASE + SEL, "agree": BASE + AGR + AGR4, "both": BASE + SEL + AGR + AGR4}
SEEDS = [int(x) for x in os.environ.get("V26_SEEDS", "0,10,20").split(",")]


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
        full = v14.full(df, tr.text[i])
        r, sents, cp = v15.rows(full, tr.text[i], s, texts[i], openings[i], refs[i])
        t2, t3, t4 = a2.get(p, ""), a3.get(p, ""), a4.get(p, "")
        r = add34(v17.add2(r, sents, t2), sents, t3, t4)
        r = add19(r, full, pk[i], texts[i], t2, openings[i])
        abs_ = [t for t in (texts[i], t2, t3, t4) if t]
        js = [jac(a, b) for a, b in combinations(abs_, 2)]
        r["agree_all"] = np.nanmean(js) if js else np.nan
        r["agree_open"] = np.nanmean([jac(a, openings[i]) for a in abs_]) if abs_ else np.nan
        built[i] = (r, sents)
    print("built", len(built), flush=True)
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

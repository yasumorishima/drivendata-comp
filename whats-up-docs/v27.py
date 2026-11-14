"""Extractive v27: v21 with wider candidates. ub.py (2026-11-14): with the true pairs, the top-40 sentences reach
0.286 and all sentences 0.369; the top 40 hold only 32% of the abstract's pairs (the paper 53%). Widening was
tried on 10-25 only with v9's sentence score (no gain); here the pair model (placement, LLM votes) ranks the
pairs of the wider pool and is trained on that pool too.

  PYTHONHASHSEED=0 V27_N=40,80,120 python whats-up-docs/v27.py cv <data_dir>   # per seed x fold, same folds
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
import v15  # noqa: E402
import v17  # noqa: E402
from extractive import rouge2  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402
from v21 import ABS4, SETS, add34, load3  # noqa: E402

NS = [int(x) for x in os.environ.get("V27_N", "40,80,120").split(",")]
SEEDS = [int(x) for x in os.environ.get("V27_SEEDS", "0,10,20").split(",")]
FEATS = SETS["v21"]


def build(tr, refs, texts, res, a2, a3, a4, n):
    v10.N_TOP = n
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    out = {}
    for i, (df, s, f) in res.items():
        p = tr.paper_id[i]
        r, sents, cp = v15.rows(v14.full(df, tr.text[i]), tr.text[i], s, texts[i], openings[i], refs[i])
        r = add34(v17.add2(r, sents, a2.get(p, "")), sents, a3.get(p, ""), a4.get(p, ""))
        out[i] = (r, sents)
    return out


def main():
    d = sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4 = v17.load2(d), load3(d), load3(d, ABS4)
    built = {n: build(tr, refs, texts, res, a2, a3, a4, n) for n in NS}
    print("built", {n: int(np.mean([len(b[i][0]) for i in b])) for n, b in built.items()}, "pairs/paper", flush=True)
    for seed in SEEDS:
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            sc = {}
            for n, b in built.items():
                m = Bag(FEATS, seed).fit(v10.with_te([b[i][0] for i in tr_ids]))
                erows = [b[i][0].copy() for i in te_ids]
                v10.target_encode([b[i][0] for i in tr_ids], erows)
                sc[n] = np.mean([rouge2(refs[i], choose_units(r, b[i][1], mix(r, m.predict(r), W)))
                                 for i, r in zip(te_ids, erows)])
            print(f"seed {seed} fold {f} " + " ".join(f"n{n} {v:.5f}" for n, v in sc.items()), flush=True)


if __name__ == "__main__":
    main()

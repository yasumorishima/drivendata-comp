"""Extractive v28: v26 (v21 + v19's evidence) with wider candidates. v27 (11-14) showed that on v21 the pair model
gains from 80-120 candidate sentences instead of 40; here the base is v26 and the pool goes further.

  PYTHONHASHSEED=0 V28_N=40,120,200 python whats-up-docs/v28.py cv <data_dir>
  python whats-up-docs/v28.py submit <data_dir> <out.csv>     # first value of V28_N (default 120)

11-14, 15 pairs vs v26: n120 +0.0011 (11/15 wins, p 0.021).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v17  # noqa: E402
import v26  # noqa: E402
from extractive import rouge2  # noqa: E402
from v5 import load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402
from v21 import ABS4, load3  # noqa: E402

NS = [int(x) for x in os.environ.get("V28_N", "120,200").split(",")]
SEEDS = [int(x) for x in os.environ.get("V28_SEEDS", "0,10,20").split(",")]
FEATS = v26.SETS["both"]


def build(tr, refs, texts, res, pk, a2, a3, a4, n):
    v10.N_TOP = n
    out = {}
    for i, (df, s, f) in res.items():
        p = tr.paper_id[i]
        out[i] = v26.feats(df, tr.text[i], s, texts[i], " ".join(str(tr.text[i]).split()[:300]), pk[i],
                           a2.get(p, ""), a3.get(p, ""), a4.get(p, ""), refs[i])
    return out


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4 = v17.load2(d), load3(d), load3(d, ABS4)
    pk = load_picks(d, tr.paper_id)
    if mode == "submit":
        built = build(tr, refs, texts, res, pk, a2, a3, a4, NS[0])  # v10.N_TOP stays NS[0] for the test papers
        os.environ.setdefault("V26_SET", "both")
        return v26.submit(d, tr, texts, built, pk, a2, a3, a4, sys.argv[3])
    for n in NS:
        b = build(tr, refs, texts, res, pk, a2, a3, a4, n)
        print(f"built n{n}", int(np.mean([len(b[i][0]) for i in b])), "pairs/paper", flush=True)
        for seed in SEEDS:
            for f in range(5):
                tr_ids = [i for i, x in res.items() if x[2] != f]
                te_ids = [i for i, x in res.items() if x[2] == f]
                m = Bag(FEATS, seed).fit(v10.with_te([b[i][0] for i in tr_ids]))
                erows = [b[i][0].copy() for i in te_ids]
                v10.target_encode([b[i][0] for i in tr_ids], erows)
                sc = np.mean([rouge2(refs[i], choose_units(r, b[i][1], mix(r, m.predict(r), W)))
                              for i, r in zip(te_ids, erows)])
                print(f"n{n} seed {seed} fold {f} {sc:.5f}", flush=True)


if __name__ == "__main__":
    main()

"""11-23 analysis on v28 (no new part): per-paper scores of v26 (n40) and v28 (n120) for 5 folds x 3 seeds, plus the
oracle upper bound (true pairs) on the same candidates, to split the model-side loss and to look at fold 2.

  PYTHONHASHSEED=0 python whats-up-docs/ana1123.py <data_dir>
Writes ~/drivendata-env/work/ana1123-papers.csv (per paper x seed x n) and ana1123-bins.csv (seed 0, n120:
predicted vs true pair rate by tf_doc / llm_votes / pos_min bins). Both stay outside the repo; only the
aggregates go to notes/.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v17  # noqa: E402
from extractive import rouge2  # noqa: E402
from v5 import load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402
from v21 import ABS4, load3  # noqa: E402
from v28 import FEATS, build  # noqa: E402

OUT = os.path.expanduser("~/drivendata-env/work")
NS = [int(x) for x in os.environ.get("A_N", "40,120").split(",")]
SEEDS = [int(x) for x in os.environ.get("A_SEEDS", "0,10,20").split(",")]


def main():
    d = sys.argv[1].rstrip("/") + "/"
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4 = v17.load2(d), load3(d), load3(d, ABS4)
    pk = load_picks(d, tr.paper_id)
    nw = {i: len(str(tr.text[i]).split()) for i in res}
    rows, bins = [], []
    for n in NS:
        b = build(tr, refs, texts, res, pk, a2, a3, a4, n)
        orc = {i: rouge2(refs[i], choose_units(b[i][0], b[i][1], b[i][0].y.values)) for i in b}
        print(f"built n{n} oracle {np.mean(list(orc.values())):.4f}", flush=True)
        for seed in SEEDS:
            for f in range(5):
                tr_ids = [i for i, x in res.items() if x[2] != f]
                te_ids = [i for i, x in res.items() if x[2] == f]
                m = Bag(FEATS, seed).fit(v10.with_te([b[i][0] for i in tr_ids]))
                erows = [b[i][0].copy() for i in te_ids]
                v10.target_encode([b[i][0] for i in tr_ids], erows)
                sc = []
                for i, r in zip(te_ids, erows):
                    p = m.predict(r)
                    s = rouge2(refs[i], choose_units(r, b[i][1], mix(r, p, W)))
                    sc.append(s)
                    rows.append({"paper": i, "fold": f, "seed": seed, "n": n, "score": s, "oracle": orc[i],
                                 "words": nw[i], "n_pairs": len(r), "pos_rate": float(r.y.mean())})
                    if seed == SEEDS[0] and n == NS[-1]:
                        bins.append(pd.DataFrame({"fold": f, "tf_doc": r.tf_doc.values, "llm_votes": r.llm_votes.values,
                                                  "pos_min": r.pos_min.values, "s_max": r.s_max.values,
                                                  "y": r.y.values, "p": p}))
                print(f"n{n} seed {seed} fold {f} {np.mean(sc):.5f}", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/ana1123-papers.csv", index=False)
    pd.concat(bins).to_csv(f"{OUT}/ana1123-bins.csv", index=False)


if __name__ == "__main__":
    main()

"""Extractive v29: v28 (v26 parts, 120 candidate sentences) plus the LLM votes of the neighbouring word pairs.

11-23 analysis (notes/whats-up-docs/stats-2026-11-23.md): the pair model is well calibrated in every bin, the loss
is ranking within a paper (AUC 0.84); 34% of the abstract's pairs occur once in the candidates (base rate 2%).
Abstracts copy runs of words, so a pair whose neighbours in the same sentence are voted by the LLM abstracts is
more likely in the abstract even when itself is not voted (e.g. a content pair inside a voted phrase).
New columns: for each occurrence, mean llm_votes of the pairs at +-1 and at +-2 (missing = 0) and the longest run
of voted pairs it sits in; per pair the max (and mean) over occurrences.

  PYTHONHASHSEED=0 python whats-up-docs/v29.py cv <data_dir>          # v28 vs v29, 5 folds x 3 seeds, same run
  python whats-up-docs/v29.py submit <data_dir> <out.csv>
"""

import os
import sys
from collections import defaultdict

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

N = 120
CTX = ["ctx1_max", "ctx1_mean", "ctx2_max", "run_max"]
SEEDS = [int(x) for x in os.environ.get("V29_SEEDS", "0,10,20").split(",")]
_feats = v26.feats


def add_ctx(r, sents):
    v = dict(zip(r.pair, r.llm_votes.fillna(0)))
    c1, c2, run = defaultdict(list), defaultdict(list), defaultdict(int)
    for t in sents.values():
        ps = v10.pairs(t)
        vs = [v.get(p, 0.0) for p in ps]
        n = len(ps)
        # length of the run of voted pairs through each position (the position itself excluded)
        left, right = [0] * n, [0] * n
        for j in range(1, n):
            left[j] = left[j - 1] + 1 if vs[j - 1] > 0 else 0
        for j in range(n - 2, -1, -1):
            right[j] = right[j + 1] + 1 if vs[j + 1] > 0 else 0
        for j, p in enumerate(ps):
            n1 = [vs[k] if 0 <= k < n else 0.0 for k in (j - 1, j + 1)]
            n2 = [vs[k] if 0 <= k < n else 0.0 for k in (j - 2, j - 1, j + 1, j + 2)]
            c1[p].append(np.mean(n1))
            c2[p].append(np.mean(n2))
            run[p] = max(run[p], left[j] + right[j])
    r["ctx1_max"] = [max(c1[p]) if c1[p] else 0.0 for p in r.pair]
    r["ctx1_mean"] = [np.mean(c1[p]) if c1[p] else 0.0 for p in r.pair]
    r["ctx2_max"] = [max(c2[p]) if c2[p] else 0.0 for p in r.pair]
    r["run_max"] = [run[p] for p in r.pair]
    return r


def feats(*a, **k):
    r, sents = _feats(*a, **k)
    return add_ctx(r, sents), sents


v26.feats = feats  # v28.build and v26.submit call v26.feats
v26.SETS["ctx"] = v26.SETS["both"] + CTX
from v28 import build  # noqa: E402

SETS = {"v28": v26.SETS["both"], "v29": v26.SETS["ctx"]}


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4 = v17.load2(d), load3(d), load3(d, ABS4)
    pk = load_picks(d, tr.paper_id)
    b = build(tr, refs, texts, res, pk, a2, a3, a4, N)  # leaves v10.N_TOP = 120 for the test papers
    print("built", len(b), flush=True)
    if mode == "submit":
        os.environ["V26_SET"] = "ctx"
        return v26.submit(d, tr, texts, b, pk, a2, a3, a4, sys.argv[3])
    for seed in SEEDS:
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([b[i][0] for i in tr_ids])
            ms = {k: Bag(fs, seed).fit(trows) for k, fs in SETS.items()}
            erows = [b[i][0].copy() for i in te_ids]
            v10.target_encode([b[i][0] for i in tr_ids], erows)
            sc = {k: [] for k in SETS}
            for i, r in zip(te_ids, erows):
                for k, m in ms.items():
                    sc[k].append(rouge2(refs[i], choose_units(r, b[i][1], mix(r, m.predict(r), W))))
            print(f"seed {seed} fold {f} " + " ".join(f"{k} {np.mean(v):.5f}" for k, v in sc.items()), flush=True)
            if seed == SEEDS[0] and f == 0:
                g = ms["v29"].ms[0].booster_.feature_importance("gain")
                print("gain:", " ".join(f"{c} {x / g.sum():.3f}" for c, x in zip(SETS["v29"], g) if c in CTX + ["llm_votes"]), flush=True)


if __name__ == "__main__":
    main()

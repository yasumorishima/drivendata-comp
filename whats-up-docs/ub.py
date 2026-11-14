"""Where are the points lost? Upper bounds of v11's unit selection (sentences + , ; : clauses, 190 words,
gain per word) when the pair probabilities are replaced by the truth (1 if the pair is in the author abstract).

  A  oracle pairs, v21's candidates (top-40 sentences by the v5 score)
  B  oracle pairs, all sentences of the paper
  C  share of the abstract's pairs that occur at all in the candidates / in the paper (recall ceiling)

  PYTHONHASHSEED=0 python whats-up-docs/ub.py <data_dir>
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import clean, rouge2  # noqa: E402
from v9 import oof  # noqa: E402
from v10 import N_TOP, pairs  # noqa: E402
from v11 import choose_units  # noqa: E402


def oracle(sents, ab):
    ps = sorted({p for s in sents.values() for p in pairs(s)})
    r = pd.DataFrame({"pair": ps})
    return choose_units(r, sents, np.array([float(p in ab) for p in ps]))


def main():
    d = sys.argv[1].rstrip("/") + "/"
    refs, texts, res = oof(d)
    out = []
    for n, (i, (df, s, f)) in enumerate(res.items()):
        ab = set(pairs(refs[i]))
        top = {k: clean(df.sent.iloc[k]) for k in np.argsort(-s)[:N_TOP]}
        alls = {k: clean(x) for k, x in enumerate(df.sent)}
        ct = {p for x in top.values() for p in pairs(x)}
        ca = {p for x in alls.values() for p in pairs(x)}
        out.append({"fold": f, "A_top40": rouge2(refs[i], oracle(top, ab)), "B_all": rouge2(refs[i], oracle(alls, ab)),
                    "rec_top40": len(ab & ct) / max(len(ab), 1), "rec_all": len(ab & ca) / max(len(ab), 1)})
        if n % 200 == 0:
            print(n, flush=True)
    o = pd.DataFrame(out)
    print(o.groupby("fold").mean().round(4))
    print(o.drop(columns="fold").mean().round(4))


if __name__ == "__main__":
    main()

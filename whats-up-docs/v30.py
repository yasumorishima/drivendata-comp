"""Extractive v30: v28 plus word-level evidence for the pair model.

11-23 analysis: 48% of the abstract's pairs get no LLM vote, and the loss is ranking within a paper. A pair the
LLM abstracts never wrote can still be made of words they did write (a voted key term next to a function word,
or a key term in a new combination). New columns, per pair (w1, w2): how many of the 4 LLM abstracts contain each
word (wv1, wv2), each word's rarity across all papers (idf1, idf2) and its tf-idf in this paper (tfidf1, tfidf2).

  PYTHONHASHSEED=0 python whats-up-docs/v30.py cv <data_dir>          # v28 vs v30, 5 folds x 3 seeds, same run
  python whats-up-docs/v30.py submit <data_dir> <out.csv>
"""

import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v17  # noqa: E402
import v26  # noqa: E402
from extractive import rouge2, tokens  # noqa: E402
from v5 import load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402
from v21 import ABS4, load3  # noqa: E402

N = 120
WORD = ["wv1", "wv2", "idf1", "idf2", "tfidf1", "tfidf2"]
SEEDS = [int(x) for x in os.environ.get("V30_SEEDS", "0,10,20").split(",")]
IDF = {}
_feats = v26.feats


def set_idf(texts):
    df = Counter()
    for t in texts:
        df.update(set(tokens(t)))
    n = len(texts)
    IDF.update({w: math.log(n / c) for w, c in df.items()})


def add_word(r, text, abstracts):
    sets = [set(tokens(a)) for a in abstracts if a]
    tf = Counter(tokens(text))
    tot = max(sum(tf.values()), 1)
    w1, w2 = [p[0] for p in r.pair], [p[1] for p in r.pair]
    for k, ws in (("1", w1), ("2", w2)):
        r["wv" + k] = [sum(w in s for s in sets) if sets else np.nan for w in ws]
        r["idf" + k] = [IDF.get(w, 0.0) for w in ws]
        r["tfidf" + k] = [1000 * tf[w] / tot * IDF.get(w, 0.0) for w in ws]
    return r


def feats(df, text, s, l1, opening, picks, t2, t3, t4, *a, **k):
    r, sents = _feats(df, text, s, l1, opening, picks, t2, t3, t4, *a, **k)
    return add_word(r, text, (l1, t2, t3, t4)), sents


v26.feats = feats  # v28.build and v26.submit call v26.feats
v26.SETS["word"] = v26.SETS["both"] + WORD
from v28 import build  # noqa: E402

SETS = {"v28": v26.SETS["both"], "v30": v26.SETS["word"]}


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    te = pd.read_csv(d + "test_features.csv")
    set_idf(tr.text.astype(str).tolist() + te.text.astype(str).tolist())
    refs, texts, res = oof(d)
    a2, a3, a4 = v17.load2(d), load3(d), load3(d, ABS4)
    pk = load_picks(d, tr.paper_id)
    b = build(tr, refs, texts, res, pk, a2, a3, a4, N)  # leaves v10.N_TOP = 120 for the test papers
    print("built", len(b), flush=True)
    if mode == "submit":
        os.environ["V26_SET"] = "word"
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
                g = ms["v30"].ms[0].booster_.feature_importance("gain")
                print("gain:", " ".join(f"{c} {x / g.sum():.3f}" for c, x in zip(SETS["v30"], g) if c in WORD + ["llm_votes"]), flush=True)


if __name__ == "__main__":
    main()

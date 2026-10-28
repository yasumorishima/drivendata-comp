"""Extractive v23: v21 with richer votes across the LLM abstracts (no new abstract).

v20-v22 showed the count of abstracts holding a word pair (llm_votes) helps, but saturates at four abstracts.
v23 changes what is counted, over the same abstracts (1: kout-llm-cpu-v1, 2: llm-abs2, 3: abs3, 4: abs4):
  vote_w        votes weighted by each abstract's hit rate (share of its candidate pairs that reach the author
                abstract, measured once on train: 4 scalars)
  tri_votes     over the pair's occurrences in the candidates, the most abstracts that hold a 3-word extension
  tri_votes_m   same, mean over occurrences
  sent_vote     best candidate sentence holding the pair, by the mean vote of that sentence's pairs
  sent_vote2    same, by the share of that sentence's pairs that are in at least two abstracts
V23_K=5 also counts the fifth abstract (abs5) in every vote.

  PYTHONHASHSEED=0 python whats-up-docs/v23.py cv <data_dir>     # v21 vs v23, same folds
  python whats-up-docs/v23.py submit <data_dir> <out.csv>
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
import v15  # noqa: E402
import v17  # noqa: E402
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table, tokens  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402
from v21 import ABS4, ALL18, E3, E4, add3, add34, load3  # noqa: E402

ABS5 = "kout-llm-abs5-v1"
K5 = os.environ.get("V23_K", "4") == "5"
NEW = ["vote_w", "tri_votes", "tri_votes_m", "sent_vote", "sent_vote2"]
V21 = ALL18 + E3 + E4 + (["llm5a", "llm5a_n", "llm5a_3"] if K5 else []) + ["llm_votes"]
SETS = {"v21": V21, "v23": V21 + NEW}
IN = ["llm", "llm2", "llm3a", "llm4a"] + (["llm5a"] if K5 else [])


def add23(r, sents, abstracts, w):
    """abstracts: list of LLM abstract texts ('' when missing), in the order of IN."""
    r["vote_w"] = sum(w[c] * r[c].fillna(0) for c in IN)
    g2 = [set(v10.pairs(a)) for a in abstracts if a]
    g3 = [v14.tri(a) for a in abstracts if a]
    tmax, tsum, occ = Counter(), Counter(), Counter()
    sv, sv2 = defaultdict(float), defaultdict(float)
    for s in sents.values():
        t = tokens(s)
        ps = list(zip(t, t[1:]))
        if not ps:
            continue
        votes = [sum(p in g for g in g2) for p in ps]
        m1, m2 = np.mean(votes), np.mean([v >= 2 for v in votes])
        for j, p in enumerate(ps):
            left = (t[j - 1],) + p if j > 0 else None
            right = p + (t[j + 2],) if j + 2 < len(t) else None
            v3 = sum(left in g or right in g for g in g3)
            tmax[p] = max(tmax[p], v3)
            tsum[p] += v3
            occ[p] += 1
            sv[p] = max(sv[p], m1)
            sv2[p] = max(sv2[p], m2)
    r["tri_votes"] = [tmax[p] for p in r.pair]
    r["tri_votes_m"] = [tsum[p] / max(occ[p], 1) for p in r.pair]
    r["sent_vote"] = [sv[p] for p in r.pair]
    r["sent_vote2"] = [sv2[p] for p in r.pair]
    return r


def add_base(r, sents, ab):
    r = add34(v17.add2(r, sents, ab[1]), sents, ab[2], ab[3])
    if K5:
        r = add3(r, sents, ab[4], "5")
        r["llm_votes"] = r.llm_votes + r.llm5a.fillna(0)
    return r


def hit_rates(rows):
    d = pd.concat(rows)
    return {c: float(d.y[d[c] == 1].gt(0).mean()) for c in IN}


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4, a5 = v17.load2(d), load3(d), load3(d, ABS4), load3(d, ABS5)
    print("abstracts 3/4/5: train", *(sum(p in a for p in tr.paper_id) for a in (a3, a4, a5)), "K5", K5, flush=True)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]

    def abst(p, llm1):
        return [llm1, a2.get(p, ""), a3.get(p, ""), a4.get(p, ""), a5.get(p, "")]

    base = {}
    for i, (df, s, f) in res.items():
        p = tr.paper_id[i]
        r, sents, cp = v15.rows(v14.full(df, tr.text[i]), tr.text[i], s, texts[i], openings[i], refs[i])
        base[i] = (add_base(r, sents, abst(p, texts[i])), sents)
    w = hit_rates([b[0] for b in base.values()])
    print("hit rates:", " ".join(f"{k} {v:.3f}" for k, v in w.items()), flush=True)
    built = {}
    for i, (r, sents) in base.items():
        p = tr.paper_id[i]
        built[i] = (add23(r, sents, abst(p, texts[i]), w), sents)
    if mode == "cv":
        sc, fold_of = {k: [] for k in SETS}, []
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([built[i][0] for i in tr_ids])
            ms = {k: Bag(fs).fit(trows) for k, fs in SETS.items()}
            erows = [built[i][0].copy() for i in te_ids]
            v10.target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                for k, m in ms.items():
                    sc[k].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, m.predict(r), W))))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
            if f == 0:
                g = ms["v23"].ms[0].booster_.feature_importance("gain")
                show = NEW + ["llm", "llm2", "llm_votes"]
                print("gain:", " ".join(f"{k} {v / g.sum():.3f}" for k, v in zip(SETS["v23"], g) if k in show), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        feats = SETS[os.environ.get("V23_SET", "v23")]
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), load_picks(d, tr.paper_id))
        te = pd.read_csv(d + "test_features.csv")
        print("abstracts 3/4/5: test", *(sum(p in a for p in te.paper_id) for a in (a3, a4, a5)), flush=True)
        tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), load_picks(d, te.paper_id))
        trn, (tst,) = add_lex(tab, [tt])
        sm = fit(trn)
        allrows = [built[i][0] for i in built]
        m = Bag(feats).fit(v10.with_te(allrows))
        summ = [""] * len(te)
        for i, df in tst.groupby("doc"):
            df = df.reset_index(drop=True)
            p = te.paper_id[i]
            s = sm.predict(df[feat_cols(df)])
            r, sents, _ = v15.rows(df, te.text[i], s, llm.get(p, ""), " ".join(str(te.text[i]).split()[:300]))
            ab = abst(p, llm.get(p, ""))
            r = add23(add_base(r, sents, ab), sents, ab, w)
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, m.predict(r), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

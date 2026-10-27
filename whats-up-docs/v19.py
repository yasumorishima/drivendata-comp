"""Extractive v19: v18 (all features, bag of 3 pair models) plus two more kinds of evidence per word pair.

  LLM-picked sentences  sel_any / sel_share / sel_rank   v5's LLM picks (6-10 of the top-30 sentences) were weak
                        as a sentence feature, but a pair repeated across picked sentences is another vote:
                        in any picked sentence / share of its occurrences in the paper that are in picked
                        sentences / best pick rank 1/(1+rank)
  agreement of the two  agree12 / agree1o / agree2o      bigram overlap (Jaccard) of the two LLM abstracts and of
  LLM abstracts         each with the opening. v17's second abstract helps in most papers and hurts where it
                        is off; with the agreement the pair model can tell when to trust it (paper-level
                        values, so they act only through interactions with llm / llm2 / llm_both)

  PYTHONHASHSEED=0 python whats-up-docs/v19.py cv <data_dir>     # v18 vs +sel vs +agree vs both, same folds
  python whats-up-docs/v19.py submit <data_dir> <out.csv>        # env V19_SET, default both
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
import v15  # noqa: E402
import v17  # noqa: E402
from extractive import BUDGET, add_lex, clean, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402

ALL18 = v14.BASE + v14.EXTRA + v17.EXTRA + v15.EXTRA
SEL = ["sel_any", "sel_share", "sel_rank"]
AGR = ["agree12", "agree1o", "agree2o"]
SETS = {"v18": ALL18, "sel": ALL18 + SEL, "agree": ALL18 + AGR, "both": ALL18 + SEL + AGR}


def jac(a, b):
    a, b = set(v10.pairs(a)), set(v10.pairs(b))
    return len(a & b) / len(a | b) if a and b else np.nan


def add19(r, df, picks, llm1, llm2, opening):
    order = {p: k for k, p in enumerate(picks)}
    occ, ins, rank = Counter(), Counter(), Counter()
    for j, s in enumerate(df.sent):
        ps = v10.pairs(clean(s))
        for p in ps:
            occ[p] += 1
            if j in order:
                ins[p] += 1
                rank[p] = max(rank[p], 1.0 / (1 + order[j]))
    has = bool(order)
    r["sel_any"] = [float(ins[p] > 0) if has else np.nan for p in r.pair]
    r["sel_share"] = [ins[p] / max(occ[p], 1) if has else np.nan for p in r.pair]
    r["sel_rank"] = [rank[p] if has else np.nan for p in r.pair]
    r["agree12"] = jac(llm1, llm2) if llm2 else np.nan
    r["agree1o"] = jac(llm1, opening)
    r["agree2o"] = jac(llm2, opening) if llm2 else np.nan
    return r


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2 = v17.load2(d)
    pk = load_picks(d, tr.paper_id)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {}
    for i, (df, s, f) in res.items():
        full = v14.full(df, tr.text[i])
        r, sents, cp = v15.rows(full, tr.text[i], s, texts[i], openings[i], refs[i])
        t2 = a2.get(tr.paper_id[i], "")
        r = v17.add2(r, sents, t2)
        built[i] = (add19(r, full, pk[i], texts[i], t2, openings[i]), sents)
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
                g = ms["both"].ms[0].booster_.feature_importance("gain")
                tot = g.sum()
                print("gain:", " ".join(f"{k} {v / tot:.3f}" for k, v in zip(SETS["both"], g) if k in SEL + AGR), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        feats = SETS[os.environ.get("V19_SET", "both")]
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), pk)
        te = pd.read_csv(d + "test_features.csv")
        tpk = load_picks(d, te.paper_id)
        tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), tpk)
        trn, (tst,) = add_lex(tab, [tt])
        sm = fit(trn)
        allrows = [built[i][0] for i in built]
        m = Bag(feats).fit(v10.with_te(allrows))
        summ = [""] * len(te)
        for i, df in tst.groupby("doc"):
            df = df.reset_index(drop=True)
            s = sm.predict(df[feat_cols(df)])
            l1, op = llm.get(te.paper_id[i], ""), " ".join(str(te.text[i]).split()[:300])
            t2 = a2.get(te.paper_id[i], "")
            r, sents, _ = v15.rows(df, te.text[i], s, l1, op)
            r = add19(v17.add2(r, sents, t2), df, tpk[i], l1, t2, op)
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, m.predict(r), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

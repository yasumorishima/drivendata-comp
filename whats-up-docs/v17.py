"""Extractive v17: v14 plus the second zero-shot LLM abstract (kout-llm-abs2-v1) as evidence in the pair model.

At sentence level (v6) the second abstract tied: the sentence score was saturated by the first one. The pair
model is a different place to use it: a pair that both LLM abstracts write is more likely the authors' wording.
Added per word pair (NaN where the paper has no second abstract: 85% of train and test have one):
  llm2 / llm2_n   in the second abstract / how often      llm2_3   share of occurrences whose 3-word extension is in it
  llm_both        in both LLM abstracts
Everything else is v14 (v11 units and mix, 190 words).

  PYTHONHASHSEED=0 python whats-up-docs/v17.py cv <data_dir>    # 5-fold CV: v14 vs v17 (same folds)
  python whats-up-docs/v17.py submit <data_dir> <out.csv>
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table, tokens  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402

ABS2 = "kout-llm-abs2-v1"
F14 = v14.BASE + v14.EXTRA
EXTRA = ["llm2", "llm2_n", "llm2_3", "llm_both"]


def load2(d):
    a = pd.read_csv(d + f"../{ABS2}/preds.csv").fillna("")
    return {p: t for p, t in zip(a.paper_id, a.llm) if len(t) > 50}


def add2(r, sents, text2):
    if not text2:
        for c in EXTRA:
            r[c] = np.nan
        return r
    g = Counter(v10.pairs(text2))
    g3 = v14.tri(text2)
    occ, hit = Counter(), Counter()
    for s in sents.values():
        t = tokens(s)
        for j in range(len(t) - 1):
            p = (t[j], t[j + 1])
            occ[p] += 1
            hit[p] += (j > 0 and (t[j - 1],) + p in g3) or (j + 2 < len(t) and p + (t[j + 2],) in g3)
    r["llm2"] = [float(p in g) for p in r.pair]
    r["llm2_n"] = [g[p] for p in r.pair]
    r["llm2_3"] = [hit[p] / max(occ[p], 1) for p in r.pair]
    r["llm_both"] = r.llm2 * r.llm
    return r


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2 = load2(d)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {}
    for i, (df, s, f) in res.items():
        r, sents, cp = v14.rows(v14.full(df, tr.text[i]), s, texts[i], openings[i], refs[i])
        built[i] = (add2(r, sents, a2.get(tr.paper_id[i], "")), sents, cp)
    if mode == "cv":
        sc, fold_of, imp = {"v14": [], "v17": []}, [], Counter()
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([built[i][0] for i in tr_ids])
            v10.FEATS[:] = F14
            m14 = v10.fit_pairs(trows)
            v10.FEATS[:] = F14 + EXTRA
            m17 = v10.fit_pairs(trows)
            imp.update(dict(zip(F14 + EXTRA, m17.booster_.feature_importance("gain"))))
            erows = [built[i][0].copy() for i in te_ids]
            v10.target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                sc["v14"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, v14.predict(m14, r, F14), W))))
                sc["v17"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, v14.predict(m17, r, F14 + EXTRA), W))))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
        tot = sum(imp.values())
        print("gain share:", " ".join(f"{k} {v / tot:.3f}" for k, v in imp.most_common(14)))
    else:
        v10.FEATS[:] = F14 + EXTRA
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), load_picks(d, tr.paper_id))
        te = pd.read_csv(d + "test_features.csv")
        tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), load_picks(d, te.paper_id))
        trn, (tst,) = add_lex(tab, [tt])
        sm = fit(trn)
        allrows = [built[i][0] for i in built]
        m = v10.fit_pairs(v10.with_te(allrows))
        summ = [""] * len(te)
        for i, df in tst.groupby("doc"):
            df = df.reset_index(drop=True)
            s = sm.predict(df[feat_cols(df)])
            r, sents, _ = v14.rows(df, s, llm.get(te.paper_id[i], ""), " ".join(str(te.text[i]).split()[:300]))
            r = add2(r, sents, a2.get(te.paper_id[i], ""))
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, v14.predict(m, r, F14 + EXTRA), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

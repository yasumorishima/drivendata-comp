"""Extractive v21: v20 plus a fourth zero-shot LLM abstract (kout-llm-abs4-v1, from the opening and the paper's
contribution statements: llm/abs4_prep.py, llm/cpu_abstract4.py). Same per-pair evidence as for the third
(llm4a / llm4a_n / llm4a_3) and llm_votes counts all four abstracts (0-4). v20 notes follow.

v20: v18 (all features, bag of 3 pair models) plus a third zero-shot LLM abstract (kout-llm-abs3-v1).

The first two LLM abstracts both read the opening and the top model sentences. The third reads the opening and
the paper's concluding part (llm/abs3_prep.py, llm/cpu_abstract3.py), where authors restate their findings in
the words they then reuse in the abstract. Added per word pair (NaN where the paper has no third abstract):
  llm3a / llm3a_n / llm3a_3   in the third abstract / how often / share of occurrences whose 3-word extension is in it
  llm_votes                   in how many of the three LLM abstracts it appears (0-3)

  PYTHONHASHSEED=0 python whats-up-docs/v21.py cv <data_dir>     # v20 vs v21, same folds
  python whats-up-docs/v21.py submit <data_dir> <out.csv>
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
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table, tokens  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402
from v18 import Bag  # noqa: E402

ABS3 = "kout-llm-abs3-v1"
ABS4 = "kout-llm-abs4-v1"
ALL18 = v14.BASE + v14.EXTRA + v17.EXTRA + v15.EXTRA
E3 = ["llm3a", "llm3a_n", "llm3a_3"]
E4 = ["llm4a", "llm4a_n", "llm4a_3"]
SETS = {"v20": ALL18 + E3 + ["llm_votes3"], "v21": ALL18 + E3 + E4 + ["llm_votes"]}


def load3(d, name=ABS3):
    a = pd.read_csv(d + f"../{name}/preds.csv").fillna("")
    return {p: t for p, t in zip(a.paper_id, a.llm) if len(t) > 50}


def add3(r, sents, text3, k="3"):
    cols = [f"llm{k}a", f"llm{k}a_n", f"llm{k}a_3"]
    if not text3:
        for c in cols:
            r[c] = np.nan
        return r
    g, g3 = Counter(v10.pairs(text3)), v14.tri(text3)
    occ, hit = Counter(), Counter()
    for s in sents.values():
        t = tokens(s)
        for j in range(len(t) - 1):
            p = (t[j], t[j + 1])
            occ[p] += 1
            hit[p] += (j > 0 and (t[j - 1],) + p in g3) or (j + 2 < len(t) and p + (t[j + 2],) in g3)
    r[cols[0]] = [float(p in g) for p in r.pair]
    r[cols[1]] = [g[p] for p in r.pair]
    r[cols[2]] = [hit[p] / max(occ[p], 1) for p in r.pair]
    return r


def add34(r, sents, t3, t4):
    r = add3(add3(r, sents, t3, "3"), sents, t4, "4")
    r["llm_votes3"] = r.llm + r.llm2.fillna(0) + r.llm3a.fillna(0)  # v20's count, for the baseline
    r["llm_votes"] = r.llm_votes3 + r.llm4a.fillna(0)
    return r


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4 = v17.load2(d), load3(d), load3(d, ABS4)
    print("abstracts 3/4: train", sum(p in a3 for p in tr.paper_id), sum(p in a4 for p in tr.paper_id), flush=True)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {}
    for i, (df, s, f) in res.items():
        r, sents, cp = v15.rows(v14.full(df, tr.text[i]), tr.text[i], s, texts[i], openings[i], refs[i])
        r = v17.add2(r, sents, a2.get(tr.paper_id[i], ""))
        built[i] = (add34(r, sents, a3.get(tr.paper_id[i], ""), a4.get(tr.paper_id[i], "")), sents)
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
                g = ms["v21"].ms[0].booster_.feature_importance("gain")
                print("gain:", " ".join(f"{k} {v / g.sum():.3f}" for k, v in zip(SETS["v21"], g) if k in E3 + E4 + ["llm", "llm2", "llm_votes"]), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        feats = SETS[os.environ.get("V21_SET", "v21")]
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), load_picks(d, tr.paper_id))
        te = pd.read_csv(d + "test_features.csv")
        print("abstracts 3/4: test", sum(p in a3 for p in te.paper_id), sum(p in a4 for p in te.paper_id), flush=True)
        tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), load_picks(d, te.paper_id))
        trn, (tst,) = add_lex(tab, [tt])
        sm = fit(trn)
        allrows = [built[i][0] for i in built]
        m = Bag(feats).fit(v10.with_te(allrows))
        summ = [""] * len(te)
        for i, df in tst.groupby("doc"):
            df = df.reset_index(drop=True)
            s = sm.predict(df[feat_cols(df)])
            r, sents, _ = v15.rows(df, te.text[i], s, llm.get(te.paper_id[i], ""), " ".join(str(te.text[i]).split()[:300]))
            r = add34(v17.add2(r, sents, a2.get(te.paper_id[i], "")), sents, a3.get(te.paper_id[i], ""), a4.get(te.paper_id[i], ""))
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, m.predict(r), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

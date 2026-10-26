"""Extractive v18: the pair model as an average of BAG LightGBMs with different seeds (less run-to-run noise),
and on top of that the candidate evidence of v15 (more placement) and v17 (second LLM abstract).

v10-v17 move about +-0.002 per fold between runs (row order / subsampling), as large as the gains being
measured. Averaging seeds removes most of it in CV and in the submission alike. CV compares, same folds:
  v14 bag   v14 features                     v17 bag   + second LLM abstract
  all bag   + second LLM abstract + v15 placement features

  PYTHONHASHSEED=0 python whats-up-docs/v18.py cv <data_dir>
  python whats-up-docs/v18.py submit <data_dir> <out.csv>     # env V18_SET (all | v17 | v14), default all
"""

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
import v15  # noqa: E402
import v17  # noqa: E402
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402

BAG = int(os.environ.get("V18_BAG", 3))
SETS = {"v14": v14.BASE + v14.EXTRA, "v17": v14.BASE + v14.EXTRA + v17.EXTRA,
        "all": v14.BASE + v14.EXTRA + v17.EXTRA + v15.EXTRA}


class Bag:
    def __init__(self, feats):
        self.feats, self.ms = feats, []

    def fit(self, rows):
        d = pd.concat(rows)
        for s in range(BAG):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=s, verbose=-1)
            self.ms.append(m.fit(d[self.feats], d.y))
        return self

    def predict(self, r):
        return np.mean([m.predict_proba(r[self.feats])[:, 1] for m in self.ms], axis=0)


def build(tr, refs, texts, res, a2):
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    out = {}
    for i, (df, s, f) in res.items():
        r, sents, cp = v15.rows(v14.full(df, tr.text[i]), tr.text[i], s, texts[i], openings[i], refs[i])
        out[i] = (v17.add2(r, sents, a2.get(tr.paper_id[i], "")), sents, cp)
    return out


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2 = v17.load2(d)
    built = build(tr, refs, texts, res, a2)
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
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        feats = SETS[os.environ.get("V18_SET", "all")]
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), load_picks(d, tr.paper_id))
        te = pd.read_csv(d + "test_features.csv")
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
            r = v17.add2(r, sents, a2.get(te.paper_id[i], ""))
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, m.predict(r), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W} bag={BAG}")


if __name__ == "__main__":
    main()

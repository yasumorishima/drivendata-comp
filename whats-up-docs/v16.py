"""Extractive v16: v14 plus word-level target encoding in the pair model.

The pair-level rate "how often this pair reaches the abstract in other papers" (te) is only informative for
pairs seen in many papers; most pairs are rare. v16 adds the same out-of-fold rate per word position:
  te1 / te2      P(pair in abstract | its first / second word), pooled over all pairs in the other papers
  te1_n / te2_n  how many pair rows that rate rests on
so a rare pair like ("we", <rare word>) or (<topic word>, "effects") borrows from its words.
Everything else is v14 (v11 units and mix, 190 words).

  PYTHONHASHSEED=0 python whats-up-docs/v16.py cv <data_dir>    # 5-fold CV: v14 vs v16 (same folds)
  python whats-up-docs/v16.py submit <data_dir> <out.csv>
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402

F14 = v14.BASE + v14.EXTRA
EXTRA = ["te1", "te2", "te1_n", "te2_n"]
_pair_te = v10.target_encode


def target_encode(train_rows, other, a=5.0):
    _pair_te(train_rows, other, a)
    hit = [Counter(), Counter()]
    n = [Counter(), Counter()]
    tot_h = tot_n = 0
    for r in train_rows:
        y = r.y.to_numpy() > 0
        for pos in (0, 1):
            w = [p[pos] for p in r.pair]
            n[pos].update(w)
            hit[pos].update(x for x, yy in zip(w, y) if yy)
        tot_h += y.sum()
        tot_n += len(y)
    prior = tot_h / max(tot_n, 1)
    for r in other:
        for pos in (0, 1):
            w = [p[pos] for p in r.pair]
            r[f"te{pos + 1}"] = [(hit[pos][x] + a * prior) / (n[pos][x] + a) for x in w]
            r[f"te{pos + 1}_n"] = [n[pos][x] for x in w]


v10.target_encode = target_encode   # with_te() looks the name up in v10 at call time


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {i: v14.rows(v14.full(df, tr.text[i]), s, texts[i], openings[i], refs[i]) for i, (df, s, f) in res.items()}
    if mode == "cv":
        sc, fold_of, imp = {"v14": [], "v16": []}, [], Counter()
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([built[i][0] for i in tr_ids])
            v10.FEATS[:] = F14
            m14 = v10.fit_pairs(trows)
            v10.FEATS[:] = F14 + EXTRA
            m16 = v10.fit_pairs(trows)
            imp.update(dict(zip(F14 + EXTRA, m16.booster_.feature_importance("gain"))))
            erows = [built[i][0].copy() for i in te_ids]
            target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                sc["v14"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, v14.predict(m14, r, F14), W))))
                sc["v16"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, v14.predict(m16, r, F14 + EXTRA), W))))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
        tot = sum(imp.values())
        print("gain share:", " ".join(f"{k} {v / tot:.3f}" for k, v in imp.most_common(12)))
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
            target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, v14.predict(m, r, F14 + EXTRA), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

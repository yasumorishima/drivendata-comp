"""Extractive v15: v14 plus more of "where in the paper" (v14's position features were the ones that helped).

Added per word pair:
  title / head   in the paper's first heading (the title) / number of headings holding it
  first10 / last10   occurrences in the first / last 10% of the paper's sentences
  para_first     share of its occurrences in the first sentence of a paragraph
  sec_i_min      lowest sentence index within its section (0 = opens a section)
  n_heads        number of distinct sections it occurs in (spread over the paper)
Everything else is v14 / v11.

  PYTHONHASHSEED=0 python whats-up-docs/v15.py cv <data_dir>    # 5-fold CV: v14 vs v15 (same folds)
  python whats-up-docs/v15.py submit <data_dir> <out.csv>
"""

import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
import v14  # noqa: E402
from extractive import BUDGET, add_lex, clean, feat_cols, fit, rouge2, table, tokens  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402

F14 = v14.BASE + v14.EXTRA
EXTRA = ["title", "head", "first10", "last10", "para_first", "sec_i_min", "n_heads"]


def rows(df, text, scores, llm_text, opening, abstract=None):
    r, sents, cp = v14.rows(df, scores, llm_text, opening, abstract)
    heads = [h.lstrip("#").strip() for h in re.findall(r"^#+ .*$", str(text), re.M)]
    title = set(v10.pairs(heads[0])) if heads else set()
    head = Counter(p for h in heads for p in set(v10.pairs(h)))
    n = len(df)
    f10, l10, pf, occ = Counter(), Counter(), Counter(), Counter()
    smin, hs = defaultdict(lambda: 99), defaultdict(set)
    for j, (s, h, si, para) in enumerate(zip(df.sent, df.heading, df.sec_i, df.para_first)):
        for p in set(v10.pairs(clean(s))):
            occ[p] += 1
            f10[p] += j < 0.1 * n
            l10[p] += j >= 0.9 * n
            pf[p] += para
            smin[p] = min(smin[p], si)
            hs[p].add(h)
    m = np.array([max(occ[p], 1) for p in r.pair], float)
    r["title"] = [float(p in title) for p in r.pair]
    r["head"] = [head[p] for p in r.pair]
    r["first10"] = [f10[p] for p in r.pair]
    r["last10"] = [l10[p] for p in r.pair]
    r["para_first"] = [pf[p] for p in r.pair] / m
    r["sec_i_min"] = [smin[p] if occ[p] else 99 for p in r.pair]
    r["n_heads"] = [len(hs[p]) for p in r.pair]
    return r, sents, cp


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {i: rows(v14.full(df, tr.text[i]), tr.text[i], s, texts[i], openings[i], refs[i])
             for i, (df, s, f) in res.items()}
    if mode == "cv":
        sc, fold_of, imp = {"v14": [], "v15": []}, [], Counter()
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([built[i][0] for i in tr_ids])
            v10.FEATS[:] = F14
            m14 = v10.fit_pairs(trows)
            v10.FEATS[:] = F14 + EXTRA
            m15 = v10.fit_pairs(trows)
            imp.update(dict(zip(F14 + EXTRA, m15.booster_.feature_importance("gain"))))
            erows = [built[i][0].copy() for i in te_ids]
            v10.target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                sc["v14"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, v14.predict(m14, r, F14), W))))
                sc["v15"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, v14.predict(m15, r, F14 + EXTRA), W))))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
        tot = sum(imp.values())
        print("gain share:", " ".join(f"{k} {v / tot:.3f}" for k, v in imp.most_common()))
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
            r, sents, _ = rows(df, te.text[i], s, llm.get(te.paper_id[i], ""), " ".join(str(te.text[i]).split()[:300]))
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, v14.predict(m, r, F14 + EXTRA), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

"""Extractive v12: v11, but the word-pair model also sees what kind of paper it is.

v10/v11's pair model lost on papers where v9 was already high (authors who copy whole sentences of the body
into the abstract): by the v9 quintile, the pair model gains +0.008 on the lowest fifth and loses -0.006 on
the highest. Per-pair features cannot tell such papers apart. v12 adds paper-level context to every pair row:
  p_top1   best v5 sentence score in the paper        p_top10  mean of the top-10 sentence scores
  s_rel    s_max / p_top1                             s_rank   rank (0 = best) of the best sentence holding the pair
Everything else (clause units, the 0.7 / 0.3 mix with v9's value, 190 words) is v11.

  python whats-up-docs/v12.py cv     <data_dir>    # 5-fold CV: v11 vs v12 (same folds, same scores)
  python whats-up-docs/v12.py submit <data_dir> <out.csv>
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402

BASE = list(v10.FEATS)
EXTRA = ["p_top1", "p_top10", "s_rel", "s_rank"]


def rows(df, scores, llm_text, opening, abstract=None):
    r, sents, cp = v10.paper_rows(df, scores, llm_text, opening, abstract)
    top = np.sort(scores)[::-1]
    rank = {k: j for j, k in enumerate(np.argsort(-scores)[:v10.N_TOP])}
    best = {}
    for k, ps in cp.items():
        for p in ps:
            best[p] = min(best.get(p, 99), rank[k])
    r["p_top1"] = top[0]
    r["p_top10"] = top[:10].mean()
    r["s_rel"] = r.s_max / max(top[0], 1e-6)
    r["s_rank"] = [best[p] for p in r.pair]
    return r, sents, cp


def predict(m, r, feats):
    return m.predict_proba(r[feats])[:, 1]


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {i: rows(df, s, texts[i], openings[i], refs[i]) for i, (df, s, f) in res.items()}
    if mode == "cv":
        sc = {"v11": [], "v12": [], "v12 w=1": []}
        fold_of = []
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([built[i][0] for i in tr_ids])
            v10.FEATS[:] = BASE
            m11 = v10.fit_pairs(trows)
            v10.FEATS[:] = BASE + EXTRA
            m12 = v10.fit_pairs(trows)
            erows = [built[i][0].copy() for i in te_ids]
            v10.target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                p11, p12 = predict(m11, r, BASE), predict(m12, r, BASE + EXTRA)
                sc["v11"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, p11, W))))
                sc["v12"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, p12, W))))
                sc["v12 w=1"].append(rouge2(refs[i], choose_units(r, built[i][1], p12)))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        v10.FEATS[:] = BASE + EXTRA
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
            r, sents, _ = rows(df, s, llm.get(te.paper_id[i], ""), " ".join(str(te.text[i]).split()[:300]))
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, predict(m, r, BASE + EXTRA), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

"""Extractive v22: v21 plus a fifth zero-shot LLM abstract (kout-llm-abs5-v1, from the opening and the starts of
the data / method and results sections: llm/abs5_prep.py, llm/cpu_abstract5.py).

v20 and v21 showed that what helps is the vote: in how many differently-prompted LLM abstracts a word pair
appears (llm_votes took 21% of the gain in v21, each abstract's own columns ~0.5%). v22 adds the fifth
abstract's columns (llm5a / llm5a_n / llm5a_3) and counts the vote over all five (llm_votes).
The v21 arm keeps its four-abstract vote (llm_votes4), so CV compares like for like.

  PYTHONHASHSEED=0 python whats-up-docs/v22.py cv <data_dir>     # v21 vs v22, same folds
  python whats-up-docs/v22.py submit <data_dir> <out.csv>
"""

import os
import sys

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
from v18 import Bag  # noqa: E402
from v20 import add3, load3  # noqa: E402

MORE = {"3": "kout-llm-abs3-v1", "4": "kout-llm-abs4-v1", "5": "kout-llm-abs5-v1"}
ALL18 = v14.BASE + v14.EXTRA + v17.EXTRA + v15.EXTRA


def cols(ks):
    return [f"llm{k}a{x}" for k in ks for x in ("", "_n", "_3")]


SETS = {"v21": ALL18 + cols("34") + ["llm_votes4"], "v22": ALL18 + cols("345") + ["llm_votes"]}


def add_more(r, sents, texts):
    for k, t in texts.items():
        r = add3(r, sents, t, k)
    r["llm_votes4"] = r.llm + r.llm2.fillna(0) + r.llm3a.fillna(0) + r.llm4a.fillna(0)
    r["llm_votes"] = r.llm_votes4 + r.llm5a.fillna(0)
    return r


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2 = v17.load2(d)
    more = {k: load3(d, n) for k, n in MORE.items()}
    print("abstracts:", {k: sum(p in a for p in tr.paper_id) for k, a in more.items()}, flush=True)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {}
    for i, (df, s, f) in res.items():
        p = tr.paper_id[i]
        r, sents, cp = v15.rows(v14.full(df, tr.text[i]), tr.text[i], s, texts[i], openings[i], refs[i])
        r = v17.add2(r, sents, a2.get(p, ""))
        built[i] = (add_more(r, sents, {k: a.get(p, "") for k, a in more.items()}), sents)
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
                g = ms["v22"].ms[0].booster_.feature_importance("gain")
                show = cols("345") + ["llm", "llm2", "llm_votes"]
                print("gain:", " ".join(f"{k} {v / g.sum():.3f}" for k, v in zip(SETS["v22"], g) if k in show), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        feats = SETS[os.environ.get("V22_SET", "v22")]
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), load_picks(d, tr.paper_id))
        te = pd.read_csv(d + "test_features.csv")
        print("abstracts test:", {k: sum(p in a for p in te.paper_id) for k, a in more.items()}, flush=True)
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
            r = add_more(v17.add2(r, sents, a2.get(p, "")), sents, {k: a.get(p, "") for k, a in more.items()})
            v10.target_encode(allrows, [r])
            summ[i] = choose_units(r, sents, mix(r, m.predict(r), W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

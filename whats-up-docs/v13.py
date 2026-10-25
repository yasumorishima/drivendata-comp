"""Extractive v13: v11 with finer units. Which pieces of a sentence may be picked?

v11 split candidate sentences at , ; : and the pair-probability greedy gained +0.0027 over whole sentences.
v13 compares two finer unit sets under the same model, mix (w=0.7) and 190 words:
  split  : also split before and / which / that / while / whereas / because / but and at parentheses
  window : v11's units + every run of 6, 10 and 14 consecutive words of the sentence (stride 2)
  python whats-up-docs/v13.py cv     <data_dir>
  python whats-up-docs/v13.py submit <data_dir> <out.csv>   # env V13_UNITS (split | window)
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11  # noqa: E402
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v10 import FEATS, fit_pairs, paper_rows, target_encode, with_te  # noqa: E402

BASE_UNITS = v11.units
SPLIT = re.compile(r"(?<=[,;:])\s+|\s+(?=(?:and|which|that|while|whereas|because|but)\b)|\s*[()]\s*")


def units_split(sents):
    out = {}
    for k, s in sents.items():
        out[(k, -1)] = s
        parts = [p.strip() for p in SPLIT.split(s) if p and p.strip()]
        if len(parts) > 1:
            for j, p in enumerate(parts):
                if len(p.split()) >= v11.MIN_W:
                    out[(k, j)] = p
    return out


def units_window(sents):
    out = BASE_UNITS(sents)
    for k, s in sents.items():
        w = s.split()
        for L in (6, 10, 14):
            if len(w) > L + 2:
                for a in range(0, len(w) - L + 1, 2):
                    out[(k, 1000 * L + a)] = " ".join(w[a:a + L])
    return out


def choose(r, sents, prob, kind):
    v11.units = {"base": BASE_UNITS, "split": units_split, "window": units_window}[kind]
    out = v11.choose_units(r, sents, prob)
    v11.units = BASE_UNITS
    return out


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {i: paper_rows(df, s, texts[i], openings[i], refs[i]) for i, (df, s, f) in res.items()}
    if mode == "cv":
        kinds = ["base", "split", "window"]
        sc, fold_of = {k: [] for k in kinds}, []
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            m = fit_pairs(with_te([built[i][0] for i in tr_ids]))
            erows = [built[i][0].copy() for i in te_ids]
            target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                q = v11.mix(r, m.predict_proba(r[FEATS])[:, 1], v11.W)
                for k in kinds:
                    sc[k].append(rouge2(refs[i], choose(r, built[i][1], q, k)))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        kind = os.environ.get("V13_UNITS", "split")
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), load_picks(d, tr.paper_id))
        te = pd.read_csv(d + "test_features.csv")
        tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), load_picks(d, te.paper_id))
        trn, (tst,) = add_lex(tab, [tt])
        sm = fit(trn)
        allrows = [built[i][0] for i in built]
        m = fit_pairs(with_te(allrows))
        summ = [""] * len(te)
        for i, df in tst.groupby("doc"):
            df = df.reset_index(drop=True)
            s = sm.predict(df[feat_cols(df)])
            r, sents, _ = paper_rows(df, s, llm.get(te.paper_id[i], ""), " ".join(str(te.text[i]).split()[:300]))
            target_encode(allrows, [r])
            summ[i] = choose(r, sents, v11.mix(r, m.predict_proba(r[FEATS])[:, 1], v11.W), kind)
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), kind)


if __name__ == "__main__":
    main()

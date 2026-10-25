"""Extractive v11: v10's word-pair probabilities, but the units to pick are clauses as well as sentences.

v9/v10 showed that picking by gain per word matters: short, dense units win. ROUGE-2 does not read grammar, so
a sentence does not have to be taken whole. Each top-40 candidate sentence (cleaned) is also split at , ; :
into clauses of at least MIN_W words; the greedy of v10 (sum of P(pair in abstract) over new distinct pairs,
per word) then chooses among sentences and clauses until 190 words. Chosen pieces are written in document
order. The pair model is v10's, unchanged (pairs that cross a split point are simply not counted).

  python whats-up-docs/v11.py cv     <data_dir>    # 5-fold CV: v9 / v10 / v11 (same folds, same scores)
  python whats-up-docs/v11.py submit <data_dir> <out.csv>   # env V11_W (default 0.7: 0.7 pair model + 0.3 v9 value)
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import BUDGET, add_lex, feat_cols, fit, rouge2, table  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof, select  # noqa: E402
from v10 import FEATS, N_TOP, choose, fit_pairs, pairs, paper_rows, target_encode, with_te  # noqa: E402

MIN_W = int(os.environ.get("V11_MIN_W", 4))
W = float(os.environ.get("V11_W", 0.7))   # CV 10-25: w=0.7 0.1910 > w=1 0.1898 > .5 0.1894 > .3 0.1878 > 0 0.1831


def units(sents):
    """(sentence key, piece index) -> text, for whole sentences (-1) and their clauses."""
    out = {}
    for k, s in sents.items():
        out[(k, -1)] = s
        parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", s)]
        if len(parts) > 1:
            for j, p in enumerate(parts):
                if len(p.split()) >= MIN_W:
                    out[(k, j)] = p
    return out


def mix(r, prob, w):
    """w * P(pair) from the pair model + (1 - w) * best v5 sentence score holding the pair (v9's value)."""
    return w * prob + (1 - w) * np.clip(r.s_max.to_numpy(), 0, None)


def choose_units(r, sents, prob, budget=BUDGET):
    p = dict(zip(r.pair, prob))
    u = units(sents)
    left = {key: ({x for x in pairs(t) if x in p}, max(len(t.split()), 1)) for key, t in u.items()}
    got, chosen, words = set(), [], 0
    while left and words < budget:
        key = max(left, key=lambda q: sum(p[x] for x in left[q][0] - got) / left[q][1])
        ps, w = left.pop(key)
        if not ps - got:
            continue
        got |= ps
        chosen.append(key)
        words += w
    # a whole sentence makes its own clauses redundant in the output
    whole = {k for k, j in chosen if j == -1}
    keep = sorted(q for q in chosen if q[1] == -1 or q[0] not in whole)
    return " ".join(u[q] for q in keep)


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {i: paper_rows(df, s, texts[i], openings[i], refs[i]) for i, (df, s, f) in res.items()}
    if mode == "cv":
        sc = {"v9": [], "v10": [], "v11": [], "v11 w=0": [], "v11 w=.3": [], "v11 w=.5": [], "v11 w=.7": []}
        fold_of = []
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            m = fit_pairs(with_te([built[i][0] for i in tr_ids]))
            erows = [built[i][0].copy() for i in te_ids]
            target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                df, s, _ = res[i]
                prob = m.predict_proba(r[FEATS])[:, 1]
                sc["v9"].append(rouge2(refs[i], select(df, s, texts[i], 0.0, 1.0, N_TOP)))
                sc["v10"].append(rouge2(refs[i], choose(r, built[i][1], built[i][2], prob)))
                sc["v11"].append(rouge2(refs[i], choose_units(r, built[i][1], prob)))
                for w in (0.0, 0.3, 0.5, 0.7):
                    sc[f"v11 w={w:g}".replace("0.", ".")].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, prob, w))))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
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
            summ[i] = choose_units(r, sents, mix(r, m.predict_proba(r[FEATS])[:, 1], W))
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

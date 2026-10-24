"""Extractive v10: v9's set selection, but each word pair gets its own probability of being in the abstract.

ROUGE-2 counts word pairs, so the unit to predict is the word pair, not the sentence. For every distinct
word pair in a paper's top-40 candidate sentences (cleaned as in the output), LightGBM predicts
P(pair in the authors' abstract) from: the best/mean v5 score of the candidate sentences containing it,
how often it occurs in the paper and in the candidates, whether the zero-shot LLM abstract has it, whether
the paper's opening has it, stop-word/number shape, and how often the pair reaches the abstract in other
papers (target encoding, out of fold). The summary is then the budgeted greedy of v9 on the sum of those
probabilities over distinct pairs, gain per word.

  python whats-up-docs/v10.py cv     <data_dir>    # 5-fold CV: v9 vs v10 (same folds, same sentence scores)
  python whats-up-docs/v10.py submit <data_dir> <out.csv>
"""

import os
import sys
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import BUDGET, add_lex, clean, feat_cols, fit, rouge2, table, tokens  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof, select  # noqa: E402

N_TOP = 40
STOP = set("the of and to in a is that for on with as by are this be we an or from it at which can not have "
           "their its these has was were also between more but they our such than how been other into".split())
FEATS = ["s_max", "s_mean", "n_cand", "tf_doc", "llm", "llm_n", "open", "stop1", "stop2", "num", "len_min", "te", "te_n"]


def pairs(s):
    t = tokens(s)
    return list(zip(t, t[1:]))


def paper_rows(df, scores, llm_text, opening, abstract=None):
    """One row per distinct word pair in the top candidates (+ which candidates hold it)."""
    order = np.argsort(-scores)[:N_TOP]
    sents = {k: clean(df.sent.iloc[k]) for k in order}
    cand_pairs = {k: set(pairs(s)) for k, s in sents.items()}
    doc = Counter(p for s in df.sent for p in pairs(clean(s)))
    g, op = Counter(pairs(llm_text)), set(pairs(opening))
    ab = set(pairs(abstract)) if abstract is not None else set()
    hold = defaultdict(list)
    for k, ps in cand_pairs.items():
        for p in ps:
            hold[p].append(scores[k])
    rows = [{"pair": p, "s_max": max(v), "s_mean": float(np.mean(v)), "n_cand": len(v), "tf_doc": doc[p],
             "llm": float(p in g), "llm_n": g[p], "open": float(p in op), "stop1": float(p[0] in STOP),
             "stop2": float(p[1] in STOP), "num": float(p[0].isdigit() or p[1].isdigit()),
             "len_min": min(len(p[0]), len(p[1])), "y": float(p in ab)} for p, v in hold.items()]
    return pd.DataFrame(rows), sents, cand_pairs


def target_encode(train_rows, other, a=5.0):
    """P(pair reaches the abstract) from the other papers; train_rows: list of per-paper frames."""
    hit, n = Counter(), Counter()
    for r in train_rows:
        n.update(r.pair)
        hit.update(r.pair[r.y > 0])
    prior = sum(hit.values()) / max(sum(n.values()), 1)
    for r in other:
        r["te"] = [(hit[p] + a * prior) / (n[p] + a) for p in r.pair]
        r["te_n"] = [n[p] for p in r.pair]


def with_te(rows):
    """Out-of-fold target encoding inside a set of papers (so the bigram model never sees its own label)."""
    rows = [r.copy() for r in rows]
    idx = np.arange(len(rows))
    for a, b in KFold(5, shuffle=True, random_state=2).split(idx):
        target_encode([rows[i] for i in a], [rows[i] for i in b])
    return rows


def fit_pairs(rows):
    d = pd.concat(rows)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    return m.fit(d[FEATS], d.y)


def choose(r, sents, cand_pairs, prob, budget=BUDGET):
    p = dict(zip(r.pair, prob))
    left = {k: (ps, max(len(sents[k].split()), 1)) for k, ps in cand_pairs.items()}
    got, chosen, words = set(), [], 0
    while left and words < budget:
        k = max(left, key=lambda k: sum(p[x] for x in left[k][0] - got) / left[k][1])
        ps, w = left.pop(k)
        if not ps - got:
            continue
        got |= ps
        chosen.append(k)
        words += w
    return " ".join(sents[k] for k in sorted(chosen))


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)   # v5 out-of-fold sentence scores (same as v9)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {i: paper_rows(df, s, texts[i], openings[i], refs[i]) for i, (df, s, f) in res.items()}
    print("pair rows", sum(len(b[0]) for b in built.values()), flush=True)
    if mode == "cv":
        v9s, v10s = [], []
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = with_te([built[i][0] for i in tr_ids])
            m = fit_pairs(trows)
            erows = [built[i][0].copy() for i in te_ids]
            target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                df, s, _ = res[i]
                v9s.append(rouge2(refs[i], select(df, s, texts[i], 0.0, 1.0, N_TOP)))
                v10s.append(rouge2(refs[i], choose(r, built[i][1], built[i][2], m.predict_proba(r[FEATS])[:, 1])))
            print("fold", f, "v9", round(np.mean(v9s), 4), "v10", round(np.mean(v10s), 4), flush=True)
        print(f"v9: {np.mean(v9s):.4f}\nv10: {np.mean(v10s):.4f}")
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
            r, sents, cp = paper_rows(df, s, llm.get(te.paper_id[i], ""), " ".join(str(te.text[i]).split()[:300]))
            target_encode(allrows, [r])
            summ[i] = choose(r, sents, cp, m.predict_proba(r[FEATS])[:, 1])
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te))


if __name__ == "__main__":
    main()

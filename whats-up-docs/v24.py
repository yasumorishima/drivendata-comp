"""Extractive + LLM sentences v24: v21's pair model, but the summary may also take sentences from the
zero-shot LLM abstracts (1: kout-llm-cpu-v1, 2: llm-abs2, 3-5: llm-abs3/4/5), not only from the paper.

v20-v23 showed that the pairs several LLM abstracts agree on are likely in the author abstract, and the
abstracts use the abstract register ("we find that", "this paper"). The paper-only selection can use those
pairs only where a candidate sentence holds them. v24 adds each LLM abstract sentence (and its , ; : clauses)
as a unit. Its pairs are scored with v21's mix (pair model + sentence score) where the paper candidates hold
them, and otherwise with the train hit rate of off-candidate pairs by vote count (how many of the abstracts
hold the pair). The greedy per-word choice is unchanged.

Arms (same pair model predictions): v21 (paper only), v24 (+ sentences of abstracts 1-5), v24s (+ only
sentences whose pairs are mostly in >= 2 abstracts, as a safety arm).

  PYTHONHASHSEED=0 python whats-up-docs/v24.py cv <data_dir>
  python whats-up-docs/v24.py submit <data_dir> <out.csv>
"""

import os
import re
import sys
from collections import Counter

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
from v11 import MIN_W, W, mix, units  # noqa: E402
from v18 import Bag  # noqa: E402
from v21 import ABS4, SETS, add34, load3  # noqa: E402

ABS5 = "kout-llm-abs5-v1"
FEATS = SETS["v21"]


def llm_units(abstracts, min_votes=0, g2=None):
    out = {}
    for a, text in enumerate(abstracts):
        for j, s in enumerate(re.split(r"(?<=[.!?])\s+", str(text).strip())):
            s = s.strip()
            if len(s.split()) < MIN_W:
                continue
            if min_votes:
                ps = v10.pairs(s)
                if not ps or np.mean([sum(p in g for g in g2) >= min_votes for p in ps]) < 0.5:
                    continue
            out[("L", a, j, -1)] = s
            parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", s)]
            if len(parts) > 1:
                for q, p in enumerate(parts):
                    if len(p.split()) >= MIN_W:
                        out[("L", a, j, q)] = p
    return out


def vote_rates(papers):
    """papers: (candidate pair set, abstracts, reference). P(off-candidate pair is in the reference | votes)."""
    hit, n = Counter(), Counter()
    for cand, abstracts, ref in papers:
        g2 = [set(v10.pairs(a)) for a in abstracts if a]
        rs = set(v10.pairs(ref))
        for p in set().union(*g2) - cand if g2 else ():
            v = sum(p in g for g in g2)
            n[v] += 1
            hit[v] += p in rs
    return {v: hit[v] / n[v] for v in n}


def choose(r, sents, score, abstracts, rates, min_votes=None):
    """Greedy per-word choice over paper units plus (unless min_votes is None) LLM abstract units."""
    p = dict(zip(r.pair, score))
    u = {("P",) + k: t for k, t in units(sents).items()}
    if min_votes is not None:
        g2 = [set(v10.pairs(a)) for a in abstracts if a]
        for x in set().union(*g2) if g2 else ():
            if x not in p:
                p[x] = rates.get(sum(x in g for g in g2), 0.0)
        u.update(llm_units(abstracts, min_votes, g2))
    left = {key: ({x for x in v10.pairs(t) if x in p}, max(len(t.split()), 1)) for key, t in u.items()}
    got, chosen, words = set(), [], 0
    while left and words < BUDGET:
        key = max(left, key=lambda q: sum(p[x] for x in left[q][0] - got) / left[q][1])
        ps, w = left.pop(key)
        if not ps - got:
            continue
        got |= ps
        chosen.append(key)
        words += w
    whole = {k[:-1] for k in chosen if k[-1] == -1}
    keep = sorted((q for q in chosen if q[-1] == -1 or q[:-1] not in whole), key=lambda q: (q[0] == "L", q[1:]))
    return " ".join(u[q] for q in keep)


ARMS = {"v21": None, "v24": 0, "v24s": 2}


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    a2, a3, a4, a5 = v17.load2(d), load3(d), load3(d, ABS4), load3(d, ABS5)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]

    def abst(p, llm1):
        return [llm1, a2.get(p, ""), a3.get(p, ""), a4.get(p, ""), a5.get(p, "")]

    built = {}
    for i, (df, s, f) in res.items():
        p = tr.paper_id[i]
        r, sents, cp = v15.rows(v14.full(df, tr.text[i]), tr.text[i], s, texts[i], openings[i], refs[i])
        r = add34(v17.add2(r, sents, a2.get(p, "")), sents, a3.get(p, ""), a4.get(p, ""))
        built[i] = (r, sents, abst(p, texts[i]))
    if mode == "cv":
        sc, fold_of, n_llm = {k: [] for k in ARMS}, [], Counter()
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            rates = vote_rates([(set(built[i][0].pair), built[i][2], refs[i]) for i in tr_ids])
            m = Bag(FEATS).fit(v10.with_te([built[i][0] for i in tr_ids]))
            erows = [built[i][0].copy() for i in te_ids]
            v10.target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                score = mix(r, m.predict(r), W)
                for k, mv in ARMS.items():
                    out = choose(r, built[i][1], score, built[i][2], rates, mv)
                    sc[k].append(rouge2(refs[i], out))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
            if f == 0:
                print("off-candidate rates by votes:", " ".join(f"{v}:{x:.3f}" for v, x in sorted(rates.items())), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
    else:
        mv = ARMS[os.environ.get("V24_ARM", "v24")]
        llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
        tab = add_sel(add_llm(table(tr.text.tolist(), tr.summary.tolist()), texts), load_picks(d, tr.paper_id))
        te = pd.read_csv(d + "test_features.csv")
        print("abstracts 2/3/4/5: test", *(sum(p in a for p in te.paper_id) for a in (a2, a3, a4, a5)), flush=True)
        tt = add_sel(add_llm(table(te.text.tolist()), [llm.get(p, "") for p in te.paper_id]), load_picks(d, te.paper_id))
        trn, (tst,) = add_lex(tab, [tt])
        sm = fit(trn)
        allrows = [built[i][0] for i in built]
        rates = vote_rates([(set(built[i][0].pair), built[i][2], refs[i]) for i in built])
        print("rates:", rates, flush=True)
        m = Bag(FEATS).fit(v10.with_te(allrows))
        summ = [""] * len(te)
        for i, df in tst.groupby("doc"):
            df = df.reset_index(drop=True)
            p = te.paper_id[i]
            s = sm.predict(df[feat_cols(df)])
            r, sents, _ = v15.rows(df, te.text[i], s, llm.get(p, ""), " ".join(str(te.text[i]).split()[:300]))
            r = add34(v17.add2(r, sents, a2.get(p, "")), sents, a3.get(p, ""), a4.get(p, ""))
            v10.target_encode(allrows, [r])
            summ[i] = choose(r, sents, mix(r, m.predict(r), W), abst(p, llm.get(p, "")), rates, mv)
        summ = [x if x.strip() else " ".join(str(t).split()[:BUDGET]) for x, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te), f"w={W}")


if __name__ == "__main__":
    main()

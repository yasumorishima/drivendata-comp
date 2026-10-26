"""Extractive v14: v11, but the word-pair model gets three new kinds of evidence.

v11's pair model sees the sentence score, counts in the paper, the zero-shot LLM abstract (same pair), the
opening and the rate in other papers. v14 adds, per pair:
  trigram  llm3 / open3   share of its occurrences in the candidates whose 3-word extension (left or right)
                          is also in the LLM abstract / the opening (the pair is used in the same phrase)
  place    pos_min / pos_mean   where in the paper it occurs (0 = start, 1 = end)
           sec_intro / sec_concl / sec_disc / sec_result / sec_method   share of its occurrences per heading type
  context  cue_max / we_max     abstract cue phrases ("this paper", "we find" ...) and we/our in the candidate
                                sentences holding it; ctx_this: share of occurrences right after this/we/our/these
Everything else (clause units, 0.7 / 0.3 mix with v9's value, 190 words) is v11.

  PYTHONHASHSEED=0 python whats-up-docs/v14.py cv <data_dir>    # 5-fold CV: v11 vs v14 (same folds, same scores)
  python whats-up-docs/v14.py submit <data_dir> <out.csv>
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10  # noqa: E402
from extractive import BUDGET, CUES, add_lex, clean, feat_cols, features, fit, rouge2, table, tokens  # noqa: E402
from v4 import LLM, add_llm  # noqa: E402
from v5 import add_sel, load_picks  # noqa: E402
from v9 import oof  # noqa: E402
from v11 import W, choose_units, mix  # noqa: E402

BASE = list(v10.FEATS)
SECS = ["intro", "concl", "disc", "result", "method"]
EXTRA = ["llm3", "open3", "pos_min", "pos_mean"] + [f"sec_{s}" for s in SECS] + ["cue_max", "we_max", "ctx_this"]
CTX = {"this", "we", "our", "these", "i"}


def tri(text):
    t = tokens(text)
    return set(zip(t, t[1:], t[2:]))


def rows(df, scores, llm_text, opening, abstract=None):
    r, sents, cp = v10.paper_rows(df, scores, llm_text, opening, abstract)
    g3, o3 = tri(llm_text), tri(opening)
    # trigram extension and context word, over occurrences in the candidates
    n_occ, l3, op3, ctx = Counter(), Counter(), Counter(), Counter()
    for s in sents.values():
        t = tokens(s)
        for j in range(len(t) - 1):
            p = (t[j], t[j + 1])
            n_occ[p] += 1
            left = (t[j - 1],) + p if j > 0 else None
            right = p + (t[j + 2],) if j + 2 < len(t) else None
            l3[p] += left in g3 or right in g3
            op3[p] += left in o3 or right in o3
            ctx[p] += j > 0 and t[j - 1] in CTX
    # place in the whole paper
    pos = defaultdict(list)
    sec = {s: Counter() for s in SECS}
    cols = {s: df[f"sec_{s}"].to_numpy() for s in SECS}
    for j, (s, ps) in enumerate(zip(df.sent, df.pos)):
        for p in set(v10.pairs(clean(s))):
            pos[p].append(ps)
            for k in SECS:
                sec[k][p] += cols[k][j]
    cue = df[[f"cue_{c.replace(' ', '_')}" for c in CUES]].sum(axis=1).to_numpy()
    we = df.has_we.to_numpy()
    cue_max, we_max = defaultdict(float), defaultdict(float)
    for k, ps in cp.items():
        for p in ps:
            cue_max[p] = max(cue_max[p], cue[k])
            we_max[p] = max(we_max[p], we[k])
    n = np.array([max(n_occ[p], 1) for p in r.pair], float)
    r["llm3"] = [l3[p] for p in r.pair] / n
    r["open3"] = [op3[p] for p in r.pair] / n
    r["ctx_this"] = [ctx[p] for p in r.pair] / n
    r["pos_min"] = [min(pos[p]) if pos[p] else 0.5 for p in r.pair]
    r["pos_mean"] = [float(np.mean(pos[p])) if pos[p] else 0.5 for p in r.pair]
    m = np.array([max(len(pos[p]), 1) for p in r.pair], float)
    for k in SECS:
        r[f"sec_{k}"] = [sec[k][p] for p in r.pair] / m
    r["cue_max"] = [cue_max[p] for p in r.pair]
    r["we_max"] = [we_max[p] for p in r.pair]
    return r, sents, cp


def full(df, text):
    """The cached v9 table keeps only sent/words; rebuild position, heading and cue columns for the paper."""
    f = features(text).reset_index(drop=True)
    assert len(f) == len(df) and (f.sent.to_numpy() == df.sent.to_numpy()).all()
    return f


def predict(m, r, feats):
    return m.predict_proba(r[feats])[:, 1]


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    refs, texts, res = oof(d)
    openings = [" ".join(str(t).split()[:300]) for t in tr.text]
    built = {i: rows(full(df, tr.text[i]), s, texts[i], openings[i], refs[i]) for i, (df, s, f) in res.items()}
    if mode == "cv":
        sc, fold_of = {"v11": [], "v14": []}, []
        imp = Counter()
        for f in range(5):
            tr_ids = [i for i, x in res.items() if x[2] != f]
            te_ids = [i for i, x in res.items() if x[2] == f]
            trows = v10.with_te([built[i][0] for i in tr_ids])
            v10.FEATS[:] = BASE
            m11 = v10.fit_pairs(trows)
            v10.FEATS[:] = BASE + EXTRA
            m14 = v10.fit_pairs(trows)
            imp.update(dict(zip(BASE + EXTRA, m14.booster_.feature_importance("gain"))))
            erows = [built[i][0].copy() for i in te_ids]
            v10.target_encode([built[i][0] for i in tr_ids], erows)
            for i, r in zip(te_ids, erows):
                sc["v11"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, predict(m11, r, BASE), W))))
                sc["v14"].append(rouge2(refs[i], choose_units(r, built[i][1], mix(r, predict(m14, r, BASE + EXTRA), W))))
                fold_of.append(f)
            fo = np.array(fold_of)
            print(f"fold {f} " + " ".join(f"{k} {np.mean(np.array(v)[fo == f]):.4f}" for k, v in sc.items()), flush=True)
        for k, v in sc.items():
            print(f"{k}: {np.mean(v):.4f}")
        tot = sum(imp.values())
        print("gain share:", " ".join(f"{k} {v / tot:.3f}" for k, v in imp.most_common()))
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

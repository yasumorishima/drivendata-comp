"""What's Up, Docs? — supervised extractive summarizer (CPU only).

Each body sentence gets a score from LightGBM trained to predict how many of its word pairs
appear in the author's abstract. The highest-scoring sentences are taken (skipping near
duplicates) until a word budget is reached, then put back in document order.

  python whats-up-docs/extractive.py cv     <data_dir>              # 5-fold CV of ROUGE-2 F1
  python whats-up-docs/extractive.py submit <data_dir> <out.csv>    # train on all, predict test

<data_dir> holds train.csv and test_features.csv (bash tools/cloud_setup.sh whats-up-docs).
"""

import re
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

BUDGET = 190  # words; the median abstract has ~180
CUES = ["this paper", "this article", "this study", "this chapter", "we argue", "we find", "we show",
        "i argue", "results show", "findings", "we propose", "this research", "in this", "suggest",
        "we examine", "we investigate", "we use", "data from", "conclude", "implications"]
SECTIONS = {"intro": r"introduc|background|overview", "concl": r"conclu|summary|final|closing",
            "disc": r"discuss|implicat", "method": r"method|data|design|sample|measure",
            "result": r"result|finding|analys", "lit": r"literature|theor|review|framework"}


# ---------- ROUGE-2 F1 (same rule as rouge_score: lower case, [a-z0-9]+ tokens, no stemming)
def tokens(x):
    return re.findall(r"[a-z0-9]+", str(x).lower())


def bigrams(x):
    t = tokens(x)
    return Counter(zip(t, t[1:]))


def rouge2(ref, pred):
    r, p = bigrams(ref), bigrams(pred)
    m = sum((r & p).values())
    if m == 0:
        return 0.0
    P, R = m / sum(p.values()), m / sum(r.values())
    return 2 * P * R / (P + R)


# ---------- sentences with their section and position
def split_doc(text):
    rows, heading, sec_i = [], "", 0
    paras = re.split(r"\n\s*\n", str(text))
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            heading = para.lstrip("#").strip().lower()
            sec_i = 0
            continue
        para = re.sub(r"\s+", " ", para)
        for j, s in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", para)):
            n = len(s.split())
            if 6 <= n <= 90 and not re.match(r"^(table|figure|fig\.)\s*\d", s.lower()):
                rows.append({"sent": s, "heading": heading, "sec_i": sec_i, "para_first": int(j == 0)})
                sec_i += 1
    return rows


def features(text):
    rows = split_doc(text)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    n = len(df)
    df["pos"] = np.arange(n) / max(n - 1, 1)
    df["idx"] = np.arange(n)
    df["rev_idx"] = n - 1 - np.arange(n)
    df["n_sent"] = n
    df["words"] = df.sent.str.split().str.len()
    low = df.sent.str.lower()
    for c in CUES:
        df["cue_" + c.replace(" ", "_")] = low.str.contains(c, regex=False).astype(int)
    for k, pat in SECTIONS.items():
        df["sec_" + k] = df.heading.str.contains(pat).astype(int)
    df["has_cite"] = low.str.contains(r"\(\D*\d{4}").astype(int)
    df["has_num"] = low.str.contains(r"\d").astype(int)
    df["has_we"] = low.str.contains(r"\b(?:we|our|i)\b").astype(int)
    # centrality: similarity to the document as a whole, and to the first sentences
    try:
        tf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(df.sent)
        cen = np.asarray(tf.mean(axis=0))
        df["sim_doc"] = (tf @ cen.T).ravel()
        lead = np.asarray(tf[: min(5, n)].mean(axis=0))
        df["sim_lead"] = (tf @ lead.T).ravel()
    except ValueError:
        df["sim_doc"] = df["sim_lead"] = 0.0
    doc_bg = bigrams(" ".join(df.sent))
    df["bg_freq"] = [np.mean([doc_bg[b] for b in bigrams(s)] or [0]) for s in df.sent]
    return df


FEATS = None


def table(docs, abstracts=None):
    out = []
    for i, text in enumerate(docs):
        df = features(text)
        if df.empty:
            continue
        df["doc"] = i
        if abstracts is not None:
            ab = bigrams(abstracts[i])
            df["y"] = [sum((bigrams(s) & ab).values()) / max(sum(bigrams(s).values()), 1) for s in df.sent]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def feat_cols(df):
    return [c for c in df.columns if c not in ("sent", "heading", "doc", "y")]


def fit(tab):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(tab[feat_cols(tab)], tab.y)
    return m


def compose(df, scores, budget=BUDGET):
    order = np.argsort(-scores)
    chosen, words, seen = [], 0, Counter()
    for k in order:
        s = df.sent.iloc[k]
        b = bigrams(s)
        if seen and sum((b & seen).values()) > 0.5 * max(sum(b.values()), 1):
            continue  # near duplicate of what is already in
        chosen.append(k)
        seen += b
        words += df.words.iloc[k]
        if words >= budget:
            break
    return " ".join(df.sent.iloc[sorted(chosen)])


def summarize(m, tab, n_docs, budget=BUDGET):
    res = [""] * n_docs
    for i, df in tab.groupby("doc"):
        res[i] = compose(df.reset_index(drop=True), m.predict(df[feat_cols(df)]), budget)
    return res


def main():
    mode, d = sys.argv[1], sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv(d + "train.csv")
    if mode == "cv":
        tab = table(tr.text.tolist(), tr.summary.tolist())
        scores = {b: [] for b in (130, 160, 190, 220)}
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
            m = fit(tab[tab.doc.isin(a)])
            te = tab[tab.doc.isin(b)]
            for bud in scores:
                summ = summarize(m, te, len(tr), bud)
                scores[bud] += [rouge2(tr.summary[i], summ[i]) for i in b]
        for bud, v in scores.items():
            print(f"budget {bud}: ROUGE-2 F1 {np.mean(v):.4f}")
    else:
        te = pd.read_csv(d + "test_features.csv")
        m = fit(table(tr.text.tolist(), tr.summary.tolist()))
        summ = summarize(m, table(te.text.tolist()), len(te))
        # fall back to the first words of the paper if no sentence survived the filter
        summ = [s if s.strip() else " ".join(str(t).split()[:BUDGET]) for s, t in zip(summ, te.text)]
        pd.DataFrame({"paper_id": te.paper_id, "summary": summ}).to_csv(sys.argv[3], index=False)
        print("wrote", sys.argv[3], len(te))


if __name__ == "__main__":
    main()

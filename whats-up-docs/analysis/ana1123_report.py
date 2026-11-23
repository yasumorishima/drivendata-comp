"""Aggregate ana1123.py output (per-paper scores outside the repo) into the numbers for notes/.
python3 whats-up-docs/analysis/ana1123_report.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

W = os.path.expanduser("~/drivendata-env/work")
p = pd.read_csv(f"{W}/ana1123-papers.csv")


def ci(d, a=.90):
    d = np.asarray(d); h = stats.t.ppf(1 - (1 - a) / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    t = stats.ttest_1samp(d, 0, alternative="greater")
    return f"{d.mean():+.4f} [{d.mean()-h:+.4f}, {d.mean()+h:+.4f}] p={t.pvalue:.3f} 勝ち {int((d > 0).sum())}/{len(d)}"


g = p.groupby(["n", "seed", "fold"]).score.mean().unstack("n")
print("[A] fold x seed の平均\n", g.round(4))
print("CV（15 対の平均）:", g.mean().round(4).to_dict())
print("[A] 15 対 n120 - n40:", ci(g[120] - g[40]))
print("    fold ごと:", (g[120] - g[40]).groupby("fold").mean().round(4).to_dict())

# per paper: average over seeds
pp = p.groupby(["n", "paper"]).agg(score=("score", "mean"), oracle=("oracle", "first"), fold=("fold", "first"),
                                     words=("words", "first")).reset_index()
w = pp.pivot(index="paper", columns="n", values="score")
meta = pp[pp.n == 40].set_index("paper")[["fold", "words"]]
meta["d"] = w[120] - w[40]
meta["s40"] = w[40]
meta["len"] = pd.qcut(meta.words, 3, labels=["短", "中", "長"])
print("\n[B] 論文の長さの区分ごと（論文 1 本 = 3 seed 平均、論文どうしの対応のある差）")
for k, x in meta.groupby("len", observed=True):
    print(f"    {k}（{int(x.words.min())}〜{int(x.words.max())} 語、{len(x)} 本）: {ci(x.d)}")
meta["lvl"] = pd.qcut(meta.s40, 3, labels=["低", "中", "高"])
print("    v26 の点の高さ別:")
for k, x in meta.groupby("lvl", observed=True):
    print(f"    {k}（{len(x)} 本）: {ci(x.d)}")
print("    fold 2 だけ 長さ別:", {str(k): round(x.d.mean(), 4) for k, x in meta[meta.fold == 2].groupby("len", observed=True)})
print("    fold 2 以外 長さ別:", {str(k): round(x.d.mean(), 4) for k, x in meta[meta.fold != 2].groupby("len", observed=True)})
print("    fold 別 論文の語数の中央値:", meta.groupby("fold").words.median().to_dict())
print("    fold 2 の差の分布: 大負け(<-0.02) %d 本・大勝ち(>0.02) %d 本／他の fold 平均 %.1f・%.1f 本" % (
    (meta[meta.fold == 2].d < -.02).sum(), (meta[meta.fold == 2].d > .02).sum(),
    (meta[meta.fold != 2].d < -.02).sum() / 4, (meta[meta.fold != 2].d > .02).sum() / 4))

o = pp.groupby("n")[["score", "oracle"]].mean()
o["model_loss"] = o.oracle - o.score
print("\n[C] 上限（正解の 2 語）とモデルの点\n", o.round(4))

b = pd.read_csv(f"{W}/ana1123-bins.csv")
print("\n[D] seed 0・n120 の 2 語ごと：区分ごとの正解率・予測の平均・区分内の AUC")
for col, edges in [("tf_doc", [0, 1, 2, 4, 8, 1e9]), ("llm_votes", [-.5, .5, 1.5, 2.5, 3.5, 9]),
                   ("pos_min", [-.01, .05, .15, .4, 1.01]), ("s_max", None)]:
    cut = pd.qcut(b[col], 5, duplicates="drop") if edges is None else pd.cut(b[col], edges)
    rows = []
    for k, x in b.groupby(cut, observed=True):
        auc = roc_auc_score(x.y, x.p) if 0 < x.y.sum() < len(x) else np.nan
        rows.append((str(k), len(x), x.y.mean(), x.p.mean(), auc, x.y.sum() / b.y.sum()))
    print(f"  {col}")
    for r in rows:
        print(f"    {r[0]:<22} 数 {r[1]:>7}  正解率 {r[2]:.3f}  予測 {r[3]:.3f}  AUC {r[4]:.3f}  正解の割合 {r[5]:.2f}")
print("  全体 AUC %.3f" % roc_auc_score(b.y, b.p))

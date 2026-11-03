"""これまでの結果の統計解析（定期的に回す）。
python3 whats-up-docs/analysis/stats.py
1) CV と LB の関係（CV を信じてよいか）
2) 実験ごとの fold 別の差の検定（揺れの外か）
3) 揺れの大きさから「出してよい差」の閾値と、必要な fold 数
"""
import csv, os
import numpy as np
from scipy import stats

D = os.path.dirname(os.path.abspath(__file__))
rows = lambda f: list(csv.DictReader(l for l in open(os.path.join(D, f)) if not l.startswith("#")))

lb = rows("lb.csv")
cv = np.array([float(r["cv"]) for r in lb]); y = np.array([float(r["lb"]) for r in lb])
r = stats.linregress(cv, y)
print(f"[1] LB = {r.slope:.3f}*CV + {r.intercept:.4f}  r={r.rvalue:.4f}  残差 sd={np.std(y-(r.slope*cv+r.intercept), ddof=2):.5f}")
print(f"    LB-CV の差 平均 {np.mean(y-cv):+.4f} sd {np.std(y-cv, ddof=1):.4f}／伸びの合計 LB/CV = {np.diff(y).sum()/np.diff(cv).sum():.2f}")

ex = rows("experiments.csv")
print("\n[2] 実験  平均差   sd     t     p(片側)  出した  LB の伸び")
nulls = []
for e in ex:
    d = np.array([float(e[f"f{i}"]) for i in range(5)])
    t = stats.ttest_1samp(d, 0, alternative="greater")
    print(f"    {e['exp']:<5} {d.mean():+.4f} {d.std(ddof=1):.4f} {t.statistic:5.2f}  {t.pvalue:.3f}   {e['shipped']}     {e['lb_gain']}")
    if t.pvalue > 0.1 and e["shipped"] == "0":
        nulls.append(d - d.mean())
pool = np.concatenate(nulls); s = np.sqrt((pool**2).sum() / (len(pool) - len(nulls)))
print(f"\n[3] 差のない実験 {len(nulls)} 本から：fold あたりの揺れ sd={s:.4f}（5 fold 平均の sd={s/np.sqrt(5):.4f}）")
for n in (5, 10, 15):
    thr = stats.t.ppf(.95, n-1) * s / np.sqrt(n); det = (stats.t.ppf(.95, n-1) + stats.t.ppf(.8, n-1)) * s / np.sqrt(n)
    print(f"    fold×seed={n:2d}: 片側 5% の閾値 {thr:.4f}／8 割の確率で見つけられる差 {det:.4f}")

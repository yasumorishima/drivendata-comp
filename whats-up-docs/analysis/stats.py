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
pred = r.slope * cv + r.intercept
print("    版   CV      LB      予想    外れ")
for row, p in zip(lb, pred):
    print(f"    {row['ver']:<4} {float(row['cv']):.4f} {float(row['lb']):.4f} {p:.4f} {float(row['lb'])-p:+.4f}")

print("\n[2] 実験  平均差   90% 区間            sd     t     p(片側)  Holm   出した  LB の伸び")
nulls, out = [], []
for e in ex:
    d = np.array([float(e[f"f{i}"]) for i in range(5)])
    t = stats.ttest_1samp(d, 0, alternative="greater")
    h = stats.t.ppf(.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    out.append((e, d, t, h))
    if t.pvalue > 0.1 and e["shipped"] == "0":
        nulls.append(d - d.mean())
ps = np.array([o[2].pvalue for o in out]); order = np.argsort(ps); holm = np.empty_like(ps); run = 0
for k, j in enumerate(order):
    run = max(run, min(1, (len(ps) - k) * ps[j])); holm[j] = run
for (e, d, t, h), hp in zip(out, holm):
    ci = f"[{d.mean()-h:+.4f}, {d.mean()+h:+.4f}]"
    print(f"    {e['exp']:<5} {d.mean():+.4f} {ci:<20} {d.std(ddof=1):.4f} {t.statistic:5.2f}  {t.pvalue:.3f}   {hp:.3f}  {e['shipped']}     {e['lb_gain']}")
pool = np.concatenate(nulls); s = np.sqrt((pool**2).sum() / (len(pool) - len(nulls)))
print(f"\n[3] 差のない実験 {len(nulls)} 本から：fold あたりの揺れ sd={s:.4f}（5 fold 平均の sd={s/np.sqrt(5):.4f}）")
for n in (5, 10, 15):
    thr = stats.t.ppf(.95, n-1) * s / np.sqrt(n); det = (stats.t.ppf(.95, n-1) + stats.t.ppf(.8, n-1)) * s / np.sqrt(n)
    print(f"    fold×seed={n:2d}: 片側 5% の閾値 {thr:.4f}／8 割の確率で見つけられる差 {det:.4f}")

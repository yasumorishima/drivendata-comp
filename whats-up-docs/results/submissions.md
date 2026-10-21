# 提出の記録（Submit Prediction が追記）

| 日時 (UTC) | commit | script | 結果 | memo |
|---|---|---|---|---|
| 2026-10-20T15:38Z | 650790d | `whats-up-docs/extractive.py` | rc=0 latest 0.1736 / best 0.1736 / rank #6 / left 2/3 | v1 教師あり抜粋（LightGBM・190語）: CV 0.170 と LB の差を測る  |
| 2026-10-20T15:55Z | 3dc8b7a | `whats-up-docs/extractive.py` | rc=0 latest 0.1789 / best 0.1789 / rank #4 / left 1/3 | v2 抜粋+言い回しモデル（単語1-2gram Ridge を LightGBM の特徴に積む・190語）: CV 0.1741  |
| 2026-10-21T02:52Z | d5ab8aa | `whats-up-docs/guided.py` | rc=0 latest 0.1822 / best 0.1822 / rank #3 / left 2/3 | v3 LLM 誘導の抜粋: v2 の文スコアの順位 + ゼロショット 7B 要旨との 2 語並び一致率の順位で選び直す。検証 100 本 0.1801→0.1866  |

# 提出の記録（Submit Prediction が追記）

| 日時 (UTC) | commit | script | 結果 | memo |
|---|---|---|---|---|
| 2026-10-20T15:38Z | 650790d | `whats-up-docs/extractive.py` | rc=0 latest 0.1736 / best 0.1736 / rank #6 / left 2/3 | v1 教師あり抜粋（LightGBM・190語）: CV 0.170 と LB の差を測る  |
| 2026-10-20T15:55Z | 3dc8b7a | `whats-up-docs/extractive.py` | rc=0 latest 0.1789 / best 0.1789 / rank #4 / left 1/3 | v2 抜粋+言い回しモデル（単語1-2gram Ridge を LightGBM の特徴に積む・190語）: CV 0.1741  |
| 2026-10-21T02:52Z | d5ab8aa | `whats-up-docs/guided.py` | rc=0 latest 0.1822 / best 0.1822 / rank #3 / left 2/3 | v3 LLM 誘導の抜粋: v2 の文スコアの順位 + ゼロショット 7B 要旨との 2 語並び一致率の順位で選び直す。検証 100 本 0.1801→0.1866  |
| 2026-10-21T07:35Z | 85b1274 | `whats-up-docs/v4.py` | rc=0 latest 0.1867 / best 0.1867 / rank #3 / left 1/3 | v4 LLM 要旨との一致（2 語並び・単語・順位）を LightGBM の特徴に学ばせる（CPU 4bit 7B のゼロショット要旨）: CV 0.1805（v2 0.1744、5 fold 全部で上）  |
| 2026-10-22T03:28Z | 25578db | `whats-up-docs/v5.py` | rc=0 latest 0.1867 / best 0.1867 / rank #3 / left 2/3 | v5 = v4 + LLM（CPU 4bit 7B）が上位 30 文から選んだ文（選ばれたか・何番目か）を特徴に: CV 0.1816（v4 0.1805、5 fold 全部で上・選びは 9 割の論文で計測）  |
| 2026-10-24T00:26Z | 559b088 | `whats-up-docs/v9.py` | rc=0 latest 0.1934 / best 0.1934 / rank #3 / left 2/3 | v9 = v5 の文の点 ＋ 組として選ぶ（上位 40 文から、2 語の並びの期待一致を重複なしで数え、1 語あたりの上積みが大きい文から 190 語まで）: CV 0.1874（v5 0.1816、5 fold 全部で上）  |
| 2026-10-25T00:38Z | b524cf3 | `whats-up-docs/v11.py` | rc= latest ? / best ? / rank ? / left ? | v11 = v10 の 2 語の並びの確率 0.7 ＋ v9 の値（文の点）0.3 を混ぜ、文と節（, ; : で切った 4 語以上）から 1 語あたりの上積みで 190 語まで: CV 0.1910（v9 0.1874、5 fold 全部で上）  |

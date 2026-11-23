# Where the Wild Features Are：作業の記録

## 2026-11-24（JST）：データの取得と、提出の経路の確認

- データ：`requests/data-cache.json`（`base_url` 付き）で data-cache の `wild-features` に入った（約 2.1 GB）。`bash tools/cloud_setup.sh wild-features`。
  - train は 541 人（音声 14,704 本）、test は 361 人。提出は `subject_id`・`outcome_score` の 2 列。
  - 生徒ごとの表：検査の種類（instrument、2 種）・年度初めの 3 課題の能力推定（theta、欠けあり）・学校 ID（train のみ）。音声ごとの表：課題・正答の文・採点の状態・採点・長さ。
- 指標は **R²**（大きいほど良い、テストの平均を出し続けると 0）。2027-02-08 は write-up の締め切り（Phase 1 の提出は 01-19 まで）。
- 提出の経路：`Submit Prediction` に `base_url` を足して K-12 のサイトに出せるようにした（ログに URL・ファイル名を出さない）。
- **v0（`wild-features/baseline.py`）：** 表だけ（音声なし）の Ridge。検査の種類・3 つの theta・課題ごとの平均の採点、音声の本数と平均の長さ。
  - CV R² 0.136（学校で分けた 5 fold。検査の種類別では 0.174・0.103）。
  - **LB 0.2836・41 位**（1 日 3 本のうち 1 本）。
  - CV と LB の差が大きい（+0.15）。学校で分けた CV が厳しすぎるのか、public のテストが偏っているのかは、まだ分からない。CV の設計（学校で分けるか）を最初に見直す。
- 次：Overview・Data ページの整理と EDA（CLAUDE.md の「コンペの進め方」）。そのあと音声の特徴を足す。

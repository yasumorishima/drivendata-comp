---
name: Training Flows
description: GPU/TPU/Colab学習フローの全体整理 — どの環境でどう学習するかの判断基準
type: reference
---

# 学習フロー一覧

## 1. Kaggle GPU フロー（GitHub Actions）

**使うとき**: 本番学習、Colab枠切れ時
**制約**: 週30時間GPU枠（GPU枠切れ→CPUフォールバック）
**対応リポ**: drivendata-comp / kaggle-competitions

```bash
gh workflow run "DrivenData GPU Train (Kaggle)" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v3 \
  -f memo="本番学習"
```

**フロー**: GH Actions → generate_notebook.py → Kaggle push → P100学習 → poll → Release → W&B sync → Discord通知
**フォールバック**: GPU枠切れ or OOM → 自動でCPU切り替え（ASR等重い学習はCPUだと非実用的）

## 2. Kaggle TPU フロー（GitHub Actions）

**使うとき**: GPU枠切れ時の代替、TPU向きモデルの学習
**制約**: 週30時間TPU枠（GPU枠とは別カウント）
**対応リポ**: drivendata-comp / signate-comp

```bash
# DrivenData
gh workflow run "DrivenData TPU Train (Kaggle)" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-tpu-v1 \
  -f memo="TPU v3-8 学習"

# SIGNATE
gh workflow run "SIGNATE TPU Train (Kaggle)" \
  --repo yasumorishima/signate-comp \
  -f competition_dir=nir-moisture \
  -f memo="TPU学習テスト"
```

**フロー**: GH Actions → generate_notebook + TPUセル注入 → Kaggle push(enable_tpu=true) → TPU v3-8学習 → poll → Release/Artifact → Discord通知
**注意**: torch_xla必須。HuggingFace TrainerはTPU自動検出するが、カスタムループは要対応

## 3. Google Colab GPU フロー（colab-mcp 新方式）

**使うとき**: 実験イテレーション、DrivenData/Kaggle等のDL学習
**制約**: Colab無料枠（1日数時間T4）
**対応リポ**: 全プロジェクト共通
**セットアップ状況**: インストール完了、OAuth認証待ち（2026-03-20）

**フロー**: Claude Code → MCP → Colab API → GPU VM自動割り当て → Jupyterカーネル経由でコード実行・結果取得

**メリット**:
- RPi5/RDP/Chromium/keepalive/Drive同期 全て不要
- Claude Codeから直接コード実行＆結果取得（experiment_runner不要）
- セッション管理がMCP側で自動化

**詳細**: `colab-experiment-setup.md` の「colab-mcp」セクション参照

### 3b. Google Colab GPU フロー（旧方式: RPi5経由）— colab-mcp移行後に廃止予定

**フロー**:
1. ローカルPC → Drive for Desktop → configをDriveに書く
2. RPi5にRDP接続 → Colabノートブックを手動実行
3. experiment_runner_v4 がDriveを監視 → train.py実行
4. 結果はDriveに保存 → ローカルPCから読む

**セッション維持**: RPi5上のCDP keepalive（20分間隔、systemdサービス）
**チェックポイント**: Colab T4は200ステップごとにチェックポイント保存 + `os.sync()` でDriveにflush

```
Drive構造:
G:\マイドライブ\kaggle\
├── runner\experiment_runner_v4.ipynb
├── runner_heartbeat.json
└── <comp>\EXP\EXP001\
    ├── train.py
    ├── config\child-exp000.yaml
    └── output\child-exp000\result.json
```

## 4. RPi5 Self-Hosted Runner

**使うとき**: CPU学習（LightGBM, scikit-learn等の軽量モデル）
**制約**: ARM64 CPU、RAM 8GB、GPU無し
**対応リポ**: signate-comp（train.yml）

```bash
gh workflow run "Train & Predict" \
  --repo yasumorishima/signate-comp \
  -f memo="LightGBM baseline"
```

**注意**: DL学習は不可。tabular/NLP（小規模）のみ

## 判断フローチャート

```
学習したい
  ├── 軽量モデル（LightGBM等）→ RPi5 Self-Hosted Runner
  ├── DLモデル（Transformers等）
  │   ├── 実験イテレーション → Colab GPU（colab-mcp）★優先
  │   ├── 本番学習
  │   │   ├── Kaggle GPU枠あり → Kaggle GPUフロー
  │   │   ├── GPU枠切れ → Colab GPU（colab-mcp）
  │   │   └── 全枠切れ → 翌日待ち（Colab: 12-24h / Kaggle GPU: 土曜）
```

## 枠リセットタイミング

| 環境 | 枠 | リセット |
|---|---|---|
| Colab 無料 T4 | 1日数時間 | 使用から12〜24時間後 |
| Kaggle GPU (P100) | 週30時間 | 毎週土曜 |
| Kaggle TPU (v3-8) | 週30時間 | 毎週土曜（GPU枠とは別） |

---

**2026-08-26 に `_archive/` へ移動。** 参照している DrivenData / Colab の枠組みが終了済み（[[_archive/drivendata-pasketti]]）。現行の判断基準は [[kaggle-methodology]] と [[feedback_compute-placement]]。

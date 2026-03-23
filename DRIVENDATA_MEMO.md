# DrivenData コンペ運用メモ

## ワークフロー一覧

| ワークフロー | 用途 | 入力 |
|---|---|---|
| Check Competitions | アクティブコンペ一覧取得 | memo |
| Download Competition Data | Playwrightでデータ自動DL→Artifact | memo |
| DrivenData Train & Validate | CSV提出型コンペの学習→提出ファイル作成 | competition_dir, memo |
| DrivenData GPU Train (Kaggle) | GPU学習：Kaggle P100で学習→Release | competition_dir, model_release_tag, memo, ... |
| Package DrivenData Submission | コード提出型：モデル+main.py→submission ZIP | competition_dir, model_release_tag, memo |

## CSV提出型（tabular）ワークフロー

```bash
gh workflow run "DrivenData Train & Validate" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=<dir名> \
  -f memo="v1: ベースライン"
```

完了後: Artifactから `submission.csv` をダウンロードして手動提出

## コード提出型（GPU）ワークフロー

GPU学習が必要なコンペ（ASR、画像認識等）は以下のフローで進める。

### フロー全体

```
1. Download Competition Data (GitHub Actions)
   → データがArtifactに保存される

2. DrivenData GPU Train (Kaggle) (GitHub Actions)
   → Kaggle P100 GPUで学習 → モデル重みをGitHub Releaseにアップロード

3. Package DrivenData Submission (GitHub Actions)
   → Releaseからモデル取得 → main.py + model/ → submission.zip作成

4. DrivenDataに手動提出
   → Artifactからsubmission.zipダウンロード → コンペページでアップロード
```

### Step 1: データダウンロード

```bash
gh workflow run "Download Competition Data" \
  --repo yasumorishima/drivendata-comp \
  -f memo="Pasketti初回DL"
```

### Step 2: GPU学習（全自動）

```bash
gh workflow run "DrivenData GPU Train (Kaggle)" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v1 \
  -f memo="v1: wav2vec2-base baseline"
```

自動で行われること:
1. `generate_notebook.py` でKaggle用ノートブック生成（train.pyをbase64でインライン埋め込み）
2. `kaggle kernels push`（P100 GPU）
3. Python API（`kernels_status`）でポーリング（push後2分待機、最大6時間）
4. kernel outputダウンロード
5. GitHub Releaseにモデルアップロード
6. W&B offline run同期
7. Discord通知

**GPU→CPUフォールバック:**
- GPUセッション上限 or クォータ切れ → 即CPUフォールバック（Discord通知）
- 実行中にGPU関連エラー（duration exceeded等） → CPUフォールバック
- 非GPU関連エラー → そのまま失敗

### Step 3: Submission ZIPパッケージング

```bash
gh workflow run "Package DrivenData Submission" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=pasketti-phonetic \
  -f model_release_tag=phonetic-model-v1 \
  -f memo="v1: baseline submission"
```

### Step 4: 手動提出

Artifactから `submission.zip` をダウンロード → DrivenDataコンペページで提出

## 新コンペ追加手順

### CSV提出型
1. `template/` をコピーして `<comp-name>/` を作成
2. `drivendata-config.json` を編集
3. `submission_format.csv` を配置
4. `train.py` を実装

### コード提出型（GPU）
1. `gpu-template/` をコピーして `<comp-name>/` を作成
2. `drivendata-config.json` を編集（submission_type: "code", artifact_name, base_model等）
3. `train.py` を実装（Kaggle GPUで実行する学習スクリプト）
4. `main.py` を実装（DrivenDataランタイムで実行する推論スクリプト）
5. `generate_notebook.py` を実装（train.pyをbase64でKaggleノートブック化）
6. `kernel-metadata.json` を作成（idはtitleのslugと一致させる）
7. `requirements-train.txt` を作成

## データDLについて

- S3公開バケットがある場合: Train & Validateワークフローが自動DL
- ない場合: Download Competition Dataワークフロー（Playwright）でDL → Artifact保存
- Kaggle kernelからの取得: GitHub Artifact → base64埋め込みノートブック内でダウンロード

## Kaggle認証（重要）

- **Secrets**: `KAGGLE_USERNAME` + `KAGGLE_KEY`（別々に設定）
- ワークフロー内で `~/.kaggle/kaggle.json` に書き出してから使用
- `KAGGLE_API_TOKEN` 環境変数だけでは認証が通らないケースあり
- `kernels_status` APIはpush直後〜起動までの間に403を返す（一時的、2分待機で回避）

## 踏んだ地雷と対策

### Kaggle output汚染（2026-03-23）
- **症状**: カーネル正常完了（2.5時間）なのに`model_*.tar.gz`が出力に見つからない
- **原因**: `data/phonetic/`（相対パス）でデータ展開 → `/kaggle/working/data/phonetic/`になり、12,000+個のflacファイルがkernel outputに含まれた
- **対策**: データは`/tmp/data/phonetic/`に展開。packaging後に`/kaggle/working/`をクリーンアップ

### Colab Drive同期失敗（2026-03-21）
- **症状**: 学習完了（CER 0.535）だが`model.safetensors`が15KBしかない
- **原因**: Colab FUSE mountが大ファイル（360MB）の同期に失敗
- **対策**: `verify_saved_model()`でファイルサイズ検証 + backupコピー + `os.sync()`強制

### Colab idle timeout（2026-03-23）
- **症状**: 学習中（step 800/3400）にランタイム切断
- **原因**: 長時間セル実行中にColab⇔ブラウザ間WebSocketがアイドル判定
- **対策**: バックグラウンドsubprocess + 60秒間隔keepaliveモニタ

### OOM on large models（2026-03-21）
- **症状**: wav2vec2-large-xlsr-53がbatch=2でもOOM
- **原因**: T4 15GBではlargeモデル+optimizer states+activationsが入らない
- **対策**: gradient_checkpointing有効化 + `get_safe_batch_size()`で自動縮小

### mask_time_prob crash（2026-03-21）
- **症状**: 短い音声でSpecAugmentがクラッシュ
- **対策**: `mask_time_prob=0.0`をモデル読み込み時に設定

## Secrets（GitHub Actions）

| Secret | 内容 |
|---|---|
| `WANDB_API_KEY` | W&B APIキー |
| `DISCORD_WEBHOOK_URL` | Discord通知 |
| `DRIVENDATA_EMAIL` | DrivenDataログイン |
| `DRIVENDATA_PASSWORD` | DrivenDataパスワード |
| `KAGGLE_USERNAME` | Kaggleユーザー名 |
| `KAGGLE_KEY` | Kaggle APIキー |
| `GH_PAT` | GitHub PAT（Kaggle kernelからArtifact DL用） |

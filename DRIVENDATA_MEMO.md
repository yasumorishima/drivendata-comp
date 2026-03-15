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

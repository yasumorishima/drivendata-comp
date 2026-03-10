# DrivenData コンペ運用メモ

## ワークフロー

```bash
gh workflow run "DrivenData Train & Validate" \
  --repo yasumorishima/drivendata-comp \
  -f competition_dir=<dir名> \
  -f memo="v1: ベースライン"
```

完了後:
1. GitHub Actions の Artifacts から `submission.csv` をダウンロード
2. https://www.drivendata.org/competitions/<id>/submissions/ から手動提出

## 新コンペ追加手順

1. `template/` をコピーして `<comp-name>/` を作成
2. `drivendata-config.json` を編集
   - `competition_id`: コンペページのURL末尾の数字
   - `s3_bucket`: S3公開バケットがあれば記載（なければ空文字）
   - `submission_format`: フォーマットCSVのファイル名
3. `submission_format.csv` をコンペページからDLして配置
4. `train.py` を実装
5. データがS3にない場合: `data/` に手動DLして `.gitignore` に追加

## データDLについて

- S3公開バケットがある場合: ワークフローが自動DL
- ない場合: コンペページから手動DL → `data/` に配置（gitignore済み）
  ```bash
  # 手動DL後はローカルで確認
  ls <comp-name>/data/
  ```

## Secrets（GitHub Actions）

| Secret | 内容 |
|---|---|
| `WANDB_API_KEY` | W&B APIキー |
| `DISCORD_WEBHOOK_URL` | Discord通知 |

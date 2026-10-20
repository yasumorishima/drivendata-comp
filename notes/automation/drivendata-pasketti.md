---
name: DrivenData On Top of Pasketti competition
description: Children's ASR competition - Word Track ($70K) + Phonetic Track ($50K), deadline 2026-04-06
type: project
---

# On Top of Pasketti: Children's Speech Recognition Challenge

## 基本情報
- **締切**: 2026-04-06 23:59 UTC
- **賞金**: $120,000（Word $70K + Phonetic $50K）
- **提出形式**: コード提出（ZIP、main.pyをルートに配置、コンテナ実行）
- **ランタイム**: Python 3.11、Docker、GPU(CUDA 12)対応
- **ランタイムリポ**: https://github.com/drivendataorg/childrens-speech-recognition-runtime
- **ベンチマークリポ**: https://github.com/drivendataorg/childrens-speech-recognition-benchmark-pub

## トラック詳細
| | Word Track (ID: 308) | Phonetic Track (ID: 309) |
|---|---|---|
| slug | childrens-word-asr | childrens-phonetic-asr |
| 評価指標 | WER (Word Error Rate) | IPA CER (Character Error Rate) |
| ベンチマーク | NVIDIA Parakeet TDT 0.6B (NeMo) | Wav2Vec2 (HuggingFace) |
| ベンチマーク精度 | WER 0.164 | CER 0.33 |
| 賞金 | 1st $25K / 2nd $15K / 3rd $10K + Noisy Bonus $5K×4 | 1st $25K / 2nd $15K / 3rd $10K |

## データ（Artifact DL済み 2026-03-14）
### Word Track（17.3 GB）
- audio_part_0/1/2.zip（各~4.2 GB、子どもの音声 .flac）
- noise_part_0/1.zip（各~2.4 GB、教室ノイズ）
- train_word_transcripts.jsonl（32 MB）

### Phonetic Track（1.4 GB）
- audio.zip（1.4 GB）
- train_phon_transcripts.jsonl（3.6 MB）

## ルール
- 外部データ/事前学習モデル: 許可（商用ライセンスのみ）
- 禁止データセット: Arizona Acoustic Child DB, CMU Kids, CSLU, MyST, RER ReadNet, RealClass, SPROUT, TalkBank(ホスト分除く), TeachFX
- 推論時ネットワーク: ブロック（モデルはZIPに同梱）
- 推論時再学習: 不可
- 勝者: MITライセンスでオープンソース化必須

## DrivenData認証
- Secrets: `DRIVENDATA_EMAIL` / `DRIVENDATA_PASSWORD`（drivendata-compリポに設定済み）
- アカウント: ユーザー名 Ymori

## ワークフロー（構築済み、3/24自動化拡張）
ほぼ全自動パイプライン:
1. `Download Competition Data` → Artifact
2. `DrivenData GPU Train (Kaggle)` → preflight → Kaggle P100学習(30min) → Release → **自動で3をトリガー**
3. `Package DrivenData Submission` → Release → submission.zip → Discord通知（提出URL付き）
4. DrivenDataに手動ZIPアップロード（API無しのため唯一の手動ステップ）

Secrets追加済み: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GH_PAT`

## 進捗（2026-03-24更新）

### Phonetic Track
- **colab-v5（wav2vec2-base, Kaggle P100）**: 5ep完了、DrivenData提出 63/65位（ベースライン）
- **v6（lv-60-espeak-cv-ft 10ep）**: Run 23484715348 実行中（3/24 19:26 JST起動、GPUリトライ版）

### Word Track
- **kernel作成済み**: `yasunorim/dd-word-asr-v1`（CLIでローカルから作成）
- **v1（export-only baseline）**: Run 23484716938 実行中（3/24 19:26 JST起動、GPUリトライ版）

### Kaggle TPU経緯
- v23-v27: OOM/idle timeout → **TPU方式断念**

### crash-proof対策（3/23実装、Drive版+リポジトリ版両方）
- gradient_checkpointing=ON（GPU時、VRAM ~40%節約）
- mask_time_prob=0.0（短音声クラッシュ防止）
- VRAM事前チェック + batch自動縮小（get_safe_batch_size）
- GC + torch.cuda.empty_cache（100step毎）
- GPUメモリログ（checkpoint保存時）
- TOKENIZERS_PARALLELISM=false / num_workers=0
- Drive版: verify_saved_model（ファイルサイズ検証+リトライ）+ backup copy
- Colabでの長時間学習: バックグラウンドsubprocess + 60秒間隔keepaliveモニタ

## 次回やること
1. **Phonetic/Word Run結果確認**（23484715348 / 23484716938）
2. **Word Track改善**: Parakeet adapter fine-tuning + noise augmentation

## 教訓
- **ignore_mismatched_sizes=True必須**: vocab_sizeが異なるモデルでCTC headサイズ不一致
- **新規kernel作成はローカルCLI push**: REST APIではslug制御不可。CLIでtitleとslugを一致させて作成
- **Kaggle GPU未割り当て問題**: enable_gpu=trueでもCPU起動する場合あり → GPUリトライ（最大3回）で対処
- **preflight checkはスクリプト化**: `scripts/preflight.py` でローカル事前検証
- **env validationを最初のセルに入れる**: CUDA互換性・データ存在をpush+poll後に気づくのは遅い
- **Drive同期確認前にcheckpoint削除しない**: final_modelのsafetensorsが同期されていなかった
- **Colab GPU制限時のポップアップは自動操作不可**: 制限到達→ポップアップ→ブラウザ操作必須→セッション死亡。resumeで対処不可。**1セッションで完走する設計が必須**
- **Colab T4で完走する設計**: epochs≤5, batch=16, grad_accum=4で30-60分。無料枠の最短GPU時間でも収まる
- **wav2vec2-lv-60-espeak-cv-ftのmask_time_prob=0.0必須**: 短い音声でクラッシュ
- **min_duration_sec=1.0でデータ27%損失**: mask_time_prob=0.0ならmin_duration不要、0.3程度で十分
- **colab-mcp接続OK**（PC側、3/23確認）: proxy modeでT4取得・セル実行・Drive mount全て動作
- **Kaggle outputにデータを置かない**: `/kaggle/working/`に展開すると全ファイルがoutputに入る。データは`/tmp/`へ

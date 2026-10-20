# LOCAL_INVENTORY

PC から cloud へ移したもの（2026-10-20・PC の Claude Code が作業）。

## 結果

- 開催中のコンペ（GEMS・SNOMED CT・What's Up Docs・練習 5 件）のメモ・データ・モデル・提出 zip は PC に無かった。
- 移したのは、学習・提出の自動化の仕組みのメモ 4 本だけ。データ（`data-cache` へ上げたもの）は 0 件。

探した場所：ホーム直下、Documents、Desktop（`Claude_code/` 配下の repo を含む）、Downloads、OneDrive、Pictures、Videos、`~/.claude` のメモ。
ファイル名でコンペ名を探したうえで、中身に `drivendata` が出てくるテキストも探した。Obsidian と Google ドライブ（G:）は PC に無かった。

## 移したもの

| 何を | どこから（PC） | どこへ |
|---|---|---|
| Pasketti の提出の流れ・ランタイム・認証 Secret の名前 | `~/.claude/projects/C--Users-fw-ya/memory/_archive/drivendata-pasketti.md` | `notes/automation/drivendata-pasketti.md` |
| 学習 workflow の呼び方と流れ | `~/.claude/projects/C--Users-fw-ya/memory/_archive/training-flows.md` | `notes/automation/training-flows.md` |
| Colab GPU 実験の構成 | `~/.claude/projects/C--Users-fw-ya/memory/_archive/colab-experiment-setup.md` | `notes/automation/colab-experiment-setup.md`（Colab の URL を伏せた） |
| コンペの進め方（共通） | `~/.claude/projects/C--Users-fw-ya/memory/kaggle-methodology.md` | `notes/automation/kaggle-methodology.md`（Colab の URL を伏せた） |

## 移さなかったもの

| 何を | どこ（PC） | 理由 |
|---|---|---|
| Discord の webhook URL だけが書かれたテキスト（137 B） | `Desktop/Kaggle/DrivenData.txt` | 秘密情報 |
| Pasketti の配布音声 350 ファイル（43 MB） | `drivendata-comp/test_log/data/phonetic/audio/` | 締切済みのコンペで、配布データ |
| 上と同じメモのコピー | `Desktop/Claude_code/claude-config/` | 重複 |
| DrivenData に触れる workflow・README | `kaggle-competitions`・`signate-comp`・プロフィール README | GitHub に既にある |

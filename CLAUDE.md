# drivendata-comp — Claude Code（cloud）向けの引き継ぎ（毎セッション最初に読む）

報告は日本語・です/ます。このファイルが作業の正本。**セッションの終わりに「現在地」と「次の一手」を更新して commit する。**
運用の細部（ワークフロー一覧・踏んだ地雷）は `DRIVENDATA_MEMO.md`、手法の説明は `README.md`。

## 作業は cloud セッションで完結させる

- user に手元の PC へ持っていかせない。user にしかできない操作（下の「PC からの受け渡し」など）だけを、1 つずつ具体的に頼む。
- 方針は `kaggle-competitions` / `signate-comp` の CLAUDE.md と同じ：
  - 提出は 1 本ずつ。結果を見て考察を `notes/` に書いてから次を決める。
  - お金をかけない（この repo は公開なので GitHub Actions の標準ランナーは無料。larger runner・有料 API は使わない）。
  - 公開手法の丸パクリはしない（土台にして改良するのは可。出典を notes に書く）。

## cloud から何が届くか（2026-10-19 実測）

| 相手 | 届くか | どうするか |
|---|---|---|
| www.drivendata.org | ❌（proxy で 403） | ログイン・DL・提出は GitHub Actions に回す（Secrets にログイン情報あり） |
| GitHub（この repo） | ✅ push・`gh api`・リリース添付の取得 | ❌ リリースの作成・削除（403）＝Actions に回す |
| GitHub Actions の dispatch | ❌（403） | **`requests/dispatch.json` を push** → `Cloud Dispatch` が代わりに起動 |
| www.kaggle.com/api/v1 | ✅（認証は proxy が付ける） | `api.kaggle.com` ではなく `www.kaggle.com/api/v1` を使う |
| Google Drive | ✅（MCP） | 今のところ DrivenData 関連のファイルは無い |

## Actions を動かす（cloud から）

```jsonc
// requests/dispatch.json を書いて push（同じ中身の再 push は動かない＝memo を変える）
{"workflow": "kaggle-train.yml",
 "inputs": {"competition_dir": "pasketti-phonetic", "model_release_tag": "phonetic-model-v6", "memo": "何を確かめるか"}}
```

- 使えるワークフローは `.github/workflows/cloud-dispatch.yml` の `ALLOWED`。
- dispatch される側のファイルは **master にも存在する必要がある**（GitHub の仕様）。新しいワークフローを作ったら master に入れてから使う。
  `data-cache.yml` は push で直接動くので master 不要。
- 状態は `gh api repos/yasumorishima/drivendata-comp/actions/runs` で読める。
- **ログは GitHub MCP の `get_job_logs`（`return_content: true`）で読む**（`gh api .../logs` は保存先の Azure が 403）。

## データ（cloud のコンテナで使う）

- 置き場所：この repo の **下書きリリース `data-cache`**（push 権限のある人にしか見えない）。添付名は `<name>__<file>`。
- 取り方：`bash tools/cloud_setup.sh [name ...]` → `~/drivendata-env/data/<name>/` に入る（分割の結合・sha256 確認つき）。
- 入れ方は 2 通り：
  1. **DrivenData から**：`requests/data-cache.json` を push（書式はファイル先頭のコメント）。Actions が Playwright で取って上げる。
  2. **user の PC から**：`bash tools/local_push.sh <name> <ファイルかフォルダ>`（下の節）。
- 🔴 **配布データは git にも公開リリースにも Artifact の外部共有にも置かない**（DrivenData の規約で再配布禁止。この repo は公開）。
  `~/drivendata-env/` の中身を repo にコピーしない。

## メモ（notes/）

- `notes/` は user と Claude の作業メモの置き場（git 管理・公開される）。PC にあったメモもここへ集める。
  - コンペごとに `notes/<コンペ>/`。実験の記録・LB・考察・次の一手を書く。
- 公開 repo なので、**ログイン情報・API キー・配布データの中身（音声・ラベルの抜き書き）は書かない。**

## PC からの受け渡し（user にお願いする操作）

PC の Claude Code に `notes/LOCAL_HANDOFF.md` の依頼文を貼ってもらう。PC 側でやること：
1. ローカルのメモ・ノートを `notes/` に集めて commit・push。
2. ローカルにだけあるデータ（DL 済みの配布物・学習済みモデル・提出 zip など）を `tools/local_push.sh` で `data-cache` に上げる。
3. 何を上げたかを `notes/LOCAL_INVENTORY.md` に書いて push。

## 現在地（2026-10-19）

- **On Top of Pasketti（子どもの音声認識）は 2026-04-06 に締切済み。** 最終順位は未記録（Phonetic は提出時点で 63/65）。
  - 最後の作業は 2026-03-24（Word Track の Kaggle GPU 学習の調整中で止まっている）。
- cloud 対応を追加（このファイル・`Cloud Dispatch`・`Data Cache`・`tools/cloud_setup.sh`・`tools/local_push.sh`・`notes/`）。
  - ✅ `Cloud Dispatch` は 10-19 に実走確認（作業ブランチから `check-competitions.yml` を起動して成功）。
  - ✅ `local_push.sh` → 下書きリリース → `cloud_setup.sh` の往復（1MB 分割つき）を 10-19 に確認。
    Actions 上（`Data Cache Selftest`）と、cloud からの取得の両方で一致。
  - ⏳ `Data Cache`（DrivenData から Playwright で取得）の初回実走を phonetic で 10-19 に起動。
- 開催中コンペ（10-19 の Check Competitions・抽出を直した版）：8 件。

  | 締切 | 分野 | 賞金 | 参加 | コンペ |
  |---|---|---|---|---|
  | 2026-12-03 | energy | $300,000 | 665 | The Geologic Enhanced Mapping System (GEMS) Prize |
  | 2027-02-01 | health | - | 176 | SNOMED CT Entity Linking Benchmark |
  | 2028-04-03 | development | - | 647 | What's Up, Docs? Document Summarization with LLMs |
  | 2028-07-03 以降 | - | - | - | 練習コンペ 5 件（Flu Shot・Conser-vision・Richter's・DengAI・Pump it Up） |

## 次の一手

1. user に PC 側の受け渡し（`notes/LOCAL_HANDOFF.md`）をお願いし、届いたメモを読んでこのファイルに反映する。
2. 開催中の DrivenData コンペを確認する：`requests/dispatch.json` に `{"workflow": "check-competitions.yml", "inputs": {"memo": "..."}}`。
3. 参加するコンペが決まったら `template/` か `gpu-template/` から `<コンペ>/` を作り、`<コンペ>/CLAUDE.md` に「🎯 上位への道筋」を書く。

## commit のルール

- モデル名は書かない。末尾に `Co-Authored-By` と `Claude-Session` の 2 行を付ける。

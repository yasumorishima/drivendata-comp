# drivendata-comp — Claude Code（cloud）向けの引き継ぎ（毎セッション最初に読む）

報告は日本語・です/ます。このファイルが作業の正本。**セッションの終わりに「現在地」と「次の一手」を更新して commit する。**
運用の細部（ワークフロー一覧・踏んだ地雷）は `DRIVENDATA_MEMO.md`、手法の説明は `README.md`。

## 🔴 user について（2026-10-20 user 指示：「わたしは日本在住、日本人　覚えておいて」）

- **user は日本在住の日本人**（米国市民・米国永住者ではない）。
- コンペを選ぶ・参加登録する前に、必ず公式ルールの参加資格をこの前提で確かめる。
  - 例：米国限定（DOE の American-Made 系など。GEMS Prize は対象外だった）、米国の制裁対象地域の除外、データ提供元の国・地域の制限。
- 「参加はできるが賞金は対象外」のコンペは、参加するかを決める前に user に伝える。

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
| CPU で LLM を回す | ✅（Actions 経由） | `requests/llm-cpu.json`（LLM on CPU：20 本並列、1 本最長 6 時間。他のワークフローは待たされる） |
| Kaggle の GPU で回す | ✅（Actions 経由） | `requests/kaggle-run.json`（Kaggle Run：入力 dataset 作成→kernel push→完了待ち→出力を data-cache `kout-<name>` へ） |
| DrivenData への参加登録・提出 | ❌ | `requests/join.json`（Join Competition）・`requests/submit.json`（Submit Prediction）を push |
| www.drivendata.org のページを読む | ❌ | **`requests/pages.json` を push** → `Fetch Pages` がログインして本文を取り、`bash tools/cloud_setup.sh pages` で読める |
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

## コンペの進め方（`notes/automation/kaggle-methodology.md` が正本・Kaggle/SIGNATE と共通）

- **戦略判断（CV 設計・指標の解釈・勝負どころ・次の一手）も Claude が自分で決める。** user に「次どうしますか」と選択肢を投げない。
- コンペを始めたら最初に：Overview と Data ページを Markdown に整理 → 公開ベースラインを読解 → 次の実験を決めるための EDA。
- 小手先の調整（ハイパラの刻み・重みの微調整）に流れない。「この改善はコンペの本質に近づいているか」を毎回問う。

## 🔴 締切済みのコンペは扱わない（2026-10-19 user 指示：「締切済みはどうでもいい」）

- 開催中のコンペだけを見る。締切済みのコンペの順位の記録・データの回収・続きの作業はしない。提案もしない。
- `pasketti-*/`（On Top of Pasketti、2026-04-06 締切）は過去の作業の跡。仕組み（GPU 学習・提出 zip・自動提出）を流用するときだけ参照する。
- PC のメモも、開催中のコンペと流用できる仕組みに関係するものだけ集めればよい。

## 現在地（2026-10-26）

- cloud 対応は 10-19〜21 に完了（`Cloud Dispatch`・`Data Cache`・`Fetch Pages`・`Join`・`Submit Prediction`・`Kaggle Run`・`LLM on CPU`）。
  - 10-22 追加：`LLM on CPU` に prep job（入力を data-cache から作り、非公開の `kin-<name>` に置く）、`Cleanup Artifacts`。
  - 🔴 **公開 repo の Actions artifact は誰でも落とせる**。論文の文を artifact に置かない（10-22 に一度置いてしまい、消す依頼を出した）。
- PC からの受け渡しは 10-20 に完了（`notes/LOCAL_INVENTORY.md`）。
  - PC の `Desktop/Kaggle/DrivenData.txt` は Discord の webhook URL（Secret `DISCORD_WEBHOOK_URL`）。Discord 側で消さない。
- **What's Up Docs：最高 LB 0.1987（3 位、v18）。** 10-26 は 2 本（v14＝2 語モデルに本文での位置・節、0.1978／v18＝2 本目の LLM 要旨も足し 3 本平均、0.1987）。10-25 は 1 本（v11、0.1953）。
  - 効いたのは「LLM（ゼロショット 7B）に要旨を書かせ、その要旨との一致で本文の文を選ぶ」。
  - 10-22 に試して効かなかったもの：LLM の文番号選び（v5）・2 本目の要旨（v6）・文埋め込み（v7）・14B。どれも CV ±0.001 以内。
  - 10-23：小さな cross-encoder を微調整して文ごとの点に（v8）→ CV +0.0003（揺れの内）。文ごとの点を足す路線は出し尽くした。
  - 10-24：**選び方を「1 語あたりの上積みが大きい文から組む」に変えたら LB +0.0067（v9）。** 2 語の並びごとの確率（v10）は CV 同点で出さず。
  - 10-25：**2 語の確率（v10）に文の点を 3 割混ぜ、文を , ; : で切った節からも選ぶと LB +0.0019（v11）。** 候補を全文に広げる・論文単位の特徴（v12）・もっと細かく切る（v13）は効かず。
  - 10-26：**2 語の確率モデルに「本文のどこに出るか」（最初の位置・序論／結論の節）を足すと LB +0.0025（v14）。** 2 本目の LLM 要旨・位置の追加・3 本平均で +0.0009（v18）。3 語一致・語ごとの率（v16）・予算は効かず。
  - 2 位（0.2021）は公開解法あり（`notes/whats-up-docs/public-2nd-botabun.md`）。有料 LLM に文を選ばせる方式。
- Kaggle の GPU 週枠（30 時間、Kaggle・SIGNATE と共通）は 10-21 に使い切った。
- 🔴 **Kaggle の GPU が使えないとき（週枠切れなど）は TPU で回す**（2026-10-22 user 指示：「kaggle GPU使えないときは、TPUで。メモ」）。
  - TPU は GPU とは別の週枠（Kaggle・SIGNATE と共通）。切り替えは kernel の `kernel-metadata.json` で `"enable_gpu": "false", "enable_tpu": "true"`。
  - TPU では CUDA 前提のもの（bitsandbytes・llama.cpp の CUDA 版・vLLM）が動かない。PyTorch/XLA か JAX/Flax で書き直す。
  - 10-23 に初めて出したら TPU の空き待ちが 1 時間以上続いた。待つ間は `TF on CPU`（Actions の CPU、fold ごと 5 本）で回す。cloud から Kaggle SDK を直接使える（`DRIVENDATA_MEMO.md`）。
- GEMS は参加資格が無く見送り（10-20 user 確認）。
- 開催中コンペ（10-26 の Check Competitions）：8 件、新規なし。

  | 締切 | 分野 | 賞金 | 参加 | コンペ |
  |---|---|---|---|---|
  | 2026-12-03 | energy | $300,000 | 735 | GEMS Prize（資格なし） |
  | 2027-02-01 | health | - | 178 | SNOMED CT Entity Linking Benchmark |
  | 2028-04-03 | development | - | 652 | What's Up, Docs? Document Summarization with LLMs |
  | 2028-07-03 以降 | - | - | - | 練習コンペ 5 件（Flu Shot・Conser-vision・Richter's・DengAI・Pump it Up） |

## 次の一手

1. **What's Up Docs（`whats-up-docs/CLAUDE.md` が正本）で毎日 3 本（UTC 0 時で回復）を 1 本ずつ出す。** 新しい部品が無い日は出さない。
   - 次の部品：v18 の土台で、2 本の LLM 要旨の一致度で効かせ方を変える・LLM が選んだ文を 2 語単位で使う。比べるときは 3 本平均（`Bag`）。
   - 出し方：`requests/submit.json` を push → `Submit Prediction` が点数を `whats-up-docs/results/submissions.md` に追記。
   - 毎日 UTC 0:07（JST 9:07）に Routine `trig_01Rp6L6ZKtha2GgqYrrDVvtn` がこのセッションを起こす。
2. SNOMED CT Benchmark：データは PhysioNet の資格審査（CITI 研修など）が要る。user に頼むかは What's Up Docs が軌道に乗ってから決める。
3. この branch を master に入れる（新しいセッションと PC が branch の切り替えなしで使えるように）。

## commit のルール

- モデル名は書かない。末尾に `Co-Authored-By` と `Claude-Session` の 2 行を付ける。

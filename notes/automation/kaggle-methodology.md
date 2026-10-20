---
name: kaggle-competition-methodology
description: 金メダル記事から学んだコンペの進め方（Kaggle/SIGNATE/DrivenData 共通）。実験管理・役割分担・開始時ルーチン・環境構成の方針。全コンペ作業時に参照必須
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 40d2a529-a80c-4095-bc37-a6f98441f119
---

# Kaggle 進め方（2026-03-16策定）

## 参考元と独自部分の区別
- **EXP + child-exp構成、役割分担（アイディアは人間/実装はAI）、EXP_SUMMARY.md、CLAUDE_COMP.md**: chimanさんの記事（https://zenn.dev/chiman/articles/b233cc808d6af3）から着想
- **コンペ開始時ルーチン・人間の確認チェックリスト**: カレーちゃん（currypurin, Kaggle GM）の note 記事「コーディングエージェント時代のKaggle──人間の役割がより本質へ」（2026-07-15、Kaggleのチュートリアル第7版ベース）から取り込み（下記 2026-07-16 節）
- **RPi5 Chromium + wtype keepalive、Google Drive同期、ファイル監視自動化、GPU-less環境設計**: 独自設計（記事とは無関係）

## 🔴 コンペ開始時ルーチン＋人間の確認チェックリスト（2026-07-16 取り込み・カレーちゃん記事、**SIGNATE 含む全コンペ適用**）

**記事の核**: エージェントは指定方向へ高速に進むが、方向が間違っていれば間違った方向へ高速に進む。放置すると hyperparam 微調整・特徴量数個追加・ensemble 重み調整等の**小手先修正に向かう**（＝CLAUDE.md の「刻み禁止・core を割る」ABSOLUTE と同旨）。人間の仕事＝**「このコンペでは何によって勝負が決まるのか」を見抜く**こと。

### 🔴🔴 本環境での適用: 「人間役」も Claude が担う（user 明示 2026-07-16）
このセットアップは**人間なしで回す**。記事で「人間が判断する」とされている仕事（CV設計 / 指標解釈 / 分布差・リークの見立て / 実験の優先度 / 結果の採否 / 勝負どころ=core の特定 / 次の一手）は**全部 Claude 自身の責任**。つまり Claude は「実装エージェント」と「戦略を考える人間」の**両方の帽子を自分でかぶり分ける**:
- 実装・実験に入る前に、必ず「人間モード」で下記チェックリストを**自分で**回し、答えを EXP_SUMMARY.md / CLAUDE_COMP.md に書き残す（放置すると自分自身が小手先修正に向かう、という記事の警告を自覚する）
- 「次どうしますか」「刻み継続か core か」を user に質問で投げない（[[feedback_kaggle]]・[[feedback_one-recommendation-not-menu]]。答えは常に core / 上位一本）
- 各実験の前後で「人間モード」の自問を挟む: **この改善はコンペの本質に近づいているか？ CV は信頼できるか？** — Yes と言えないまま実験数だけ積まない
- user の役割は目標設定（上位一本）と承認のみ。戦略判断を user に外注しない

### コンペ開始時ルーチン（毎コンペ必ず）
1. **Overview + Data ページをエージェントに渡して整理**: 目的・train/test 構造・目的変数・評価指標・提出形式・注意ルール → **Markdown ファイルに保存**（docs/ 配下、CLAUDE_COMP.md の素材）
2. **公開ベースライン Notebook を「読解」させる**: ただ動かすのでなく、何をしているか・どの処理が重要か・どこを変更できそうかを説明させる
3. **「次の実験を決めるための EDA」を実行**: きれいなグラフ量産ではなく、CV をどう組むか / train-test の違い / グループ・時系列の考慮要否 / 重要特徴量 / モデルがどこで間違えるか / 指標に合わせた改善余地、を出す

### 人間が確認すべきチェックリスト（戦略が変わる基礎情報）
- 評価指標は**実際に何を測っているか**（Accuracy か順位相関か確率精度か、難サンプル 1 個の寄与か）→ 改善の方向・後処理の方針が変わる
- **CV は Public LB をどの程度再現できているか**（信頼できない CV のまま大量実験は無意味＝CV↑LB↓ / Public↑Private↓ の温床）
- train と test に**分布差**はあるか
- **時系列・グループ・重複データ**の考慮要否（同一ユーザー/患者/製品が複数行 → グループまたぎ分割禁止、時系列 → 未来情報リーク禁止）
- **リーク**の可能性
- モデルは**どのようなサンプルで間違えているか**
- Discussion の重要情報は何か
- **今の改善はコンペの本質（core）に近づいているか**（勝負どころ候補: CV設計 / 分布差対応 / 外部データ・事前学習 / 後処理×指標 / ラベル生成過程・データ生成過程の理解）

### エージェントに任せる / 人間が判断する（分担の再確認）
- 任せる: コード作成・Notebook 読解・Discussion 整理・EDA 実行・論文/手法調査・実験実行・結果レポート化・リファクタ/エラー修正
- 人間: CV 設計 / 指標解釈 / データを見る視点 / 実験の優先度 / 結果の採否 / 勝負どころの特定 / 次の一手

## 原則0：全コンペこの構成で統一する
- **Kaggle / SIGNATE / DrivenData 問わず**、全コンペに EXP + child-exp 構成を適用（user 明示 2026-07-16: SIGNATE にも本 methodology 全体＝開始時ルーチン・チェックリスト含む＝を適用）
- Drive上の `G:/マイドライブ/kaggle/<comp-slug>/` にEXP構成を作成
- 新しいコンペに参加する際は、まずDriveにフォルダ構成 + EXP_SUMMARY.md + CLAUDE_COMP.md を作成してから着手

## 原則1：「広く浅く」は禁止、「狭く深く」で勝つ

- **同時参加は最大2コンペ**。1コンペに最低50実験は回す覚悟で臨む
- 実験回数が足りないコンペにsubmitしても順位は上がらない
- 「とりあえず出す」は時間の無駄。出すなら仮説を持って出す

## 役割分担：アイディアは人間、実装はAI
> **⚠️ 2026-07-16 再定義: 本環境の「人間」= Claude 自身**（user 明示「我々の場合は人間なしでやる」）。以下の「人間がやること」は user を待つ項目ではなく、**Claude が人間モードとして自分で果たす責務**。詳細は上の「本環境での適用」節。

### 人間がやること（AIに任せない）
- **仮説を立てる**: データを観察し、「なぜこのサンプルで誤差が大きいか」を考える
- **何を試すか決める**: 施策の優先度を判断する
- **ドメイン知識を入れる**: コンペ固有の工夫はAIからは出てこない
- **CV設計を決める**: GroupKFoldの切り方、Stratifyの軸など
- **結果を解釈する**: OOF分析の結果から次の一手を考える

### AIに任せること
- **実装**: train.py、特徴量エンジニアリング、モデルコード
- **分析**: OOFエラー分析、CV-LB相関分析、カテゴリ別誤差可視化
- **公開Notebook/Discussionの要約**: 大量の情報を整理する
- **実験管理**: config生成、結果記録、EXP_SUMMARY.md更新

### AIに「精度上げろ」と丸投げしない
- AIは教科書的な施策（アンサンブル、TTA、正則化）しか出せない
- 「このSpeciesが苦手だから改善して」→ loss重み変更 → CV↑LB↓ のループに陥る
- **AIには分析させ、その結果を人間が読んでアイディアを出す**

## 実験管理構成：EXP + child-exp

```
kaggle-competitions/<comp-slug>/
├── CLAUDE_COMP.md              # コンペ固有のガードレール
├── EXP_SUMMARY.md              # 実験履歴（AIの記憶）
├── docs/
│   └── Idea_Research/          # アイディアメモ・Deep Research結果
├── EXP/
│   ├── EXP001/
│   │   ├── train.py            # メインパイプライン
│   │   ├── infer.py            # 推論
│   │   ├── config/
│   │   │   ├── child-exp000.yaml  # ベース
│   │   │   ├── child-exp001.yaml  # 特徴量追加
│   │   │   └── child-exp002.yaml  # loss変更
│   │   └── output/
│   │       ├── child-exp000/
│   │       │   ├── oof.csv     # Out-of-Fold予測
│   │       │   ├── train.log
│   │       │   └── result.json # CVスコア
│   │       └── ...
│   └── EXP002/                 # パイプライン大幅変更時
└── submit/                     # 最終提出用Kaggle kernel
```

### EXPの使い分け
- **child-exp**: パラメータ、loss、特徴量の変更 → 同じEXP内
- **新EXP**: train.pyのパイプラインを大きく変えるとき

### EXP_SUMMARY.md の書き方
```markdown
| EXP | child | 変更内容 | CV | LB | 結果 | 所感 |
|-----|-------|---------|-----|-----|------|------|
| 001 | 000 | baseline (LightGBM) | 0.890 | 0.885 | ベース | — |
| 001 | 001 | + target encoding | 0.895 | 0.890 | ↑ | 効いた |
| 001 | 002 | + focal loss | 0.900 | 0.882 | CV↑LB↓ | 過学習 |
```

### EXP_SUMMARY.mdの注意点
- 失敗実験を大量に読ませるとAIが消極的になる（「もう打ち手がない」と言い出す）
- **ガードレール（同じ失敗を繰り返させない）として使い、次のアイディアは人間が出す**
- LBスコアは手入力（API取得で自動化も可）

## CLAUDE_COMP.md（コンペ固有ガードレール）

コンペごとに作成。内容例：
- 評価関数の正しい実装と間違った実装の両方を明記
- 「CV↑LB↓になった施策」リスト（二度と試さない）
- infer.pyのよくあるバグ（変数名ミス等）
- 後方互換性ルール（child-expで既存configを壊さない）
- 禁止事項（Trainの分布を歪める系の施策 etc.）

## 実験環境

### 制約
- ローカル: Celeron N4500 / 4GB RAM — **ML学習実行禁止**（メモリ不足）
- RPi5: Cortex-A76 4コア 2.4GHz — 24/7稼働、Playwright稼働中
- Kaggle kernel: 9h GPU / 12h CPU — 最終提出専用
- Colab無料枠: 90分アイドルタイムアウト / 12時間最大セッション

### 本番構成：Google Drive + Colab + RPi5

```
[ローカル PC]                           [Google Colab (無料)]
Claude Code                              ファイル監視Notebook
  ↓ config/train.py作成                    ↓ 自動学習実行
  ↓                                       ↓
Google Drive (for Desktop) ←――――――――→ Google Drive (mount)
  EXP/config/child-exp005.yaml           新config検知→train.py実行
  EXP/output/child-exp005/result.json    結果をDriveに保存
```

#### 各コンポーネントの役割

**ローカルPC（Claude Code）**
- コード作成・編集（train.py, config.yaml）→ Driveフォルダに直接書く
- 結果をDriveから読んで分析
- EXP_SUMMARY.md更新

**Google Colab（学習実行）**
- ユーザーが普段のブラウザで1日1回開いてセル実行（手動）
- ファイル監視の`while True`ループが回り続ける → **アイドルタイムアウトは発生しない**
- 新config検知 → 自動でtrain.py実行 → 結果をDriveに保存
- 12時間の最大セッション制限で切れたら、翌日また開く
- テーブルデータ: CPU / 画像・NLP: GPU

**RPi5（テーブルデータの高速実行）**
- LightGBM/XGBoostはARM64対応、8GB RAMで十分
- SSH経由でClaude Codeから直接train.py実行可能
- Colabの12h制限なし、反復速度最速
- GPUが不要なコンペはRPi5で完結

**Kaggle kernel**
- 最終提出専用（開発には使わない）
- 実験で確定したベストconfigをsubmit用notebookに反映

#### Colabファイル監視スクリプト
Colab URL: <Colab notebook URL (removed)>
Drive上: `G:/マイドライブ/kaggle/kaggle_experiment_runner.ipynb`

while Trueループが実行中である限り、Colabセッションはアイドルにならない。
RPi5によるkeepaliveは不要。12h最大セッション制限のみ手動で対応（1日1回再実行）。

### コンペ種別ごとの使い分け

| コンペ種別 | 開発サイクル | 最終提出 |
|---|---|---|
| テーブルデータ | **RPi5直接実行**（最速） or Colab CPU | Kaggle kernel |
| 画像・NLP（GPU必須） | Colab GPU（RPi5 keepaliveで維持） | Kaggle kernel |
| DrivenData（GPU必須） | Colab GPU（同じ構成） | DrivenData Submit |

**DrivenDataでも同じ構成を使う。** Drive上の`kaggle/`フォルダにコンペごとのEXPフォルダを作成し、Colabノートブックの`COMP`変数を切り替えるだけ。

## 実験サイクル（1日の流れ）

1. **朝**: 前日の実験結果を確認（output/result.json）
2. **分析**: AIにOOF分析させる（カテゴリ別誤差、CV-LB相関）
3. **仮説**: 分析結果を見て人間がアイディアを出す
4. **実装**: AIにchild-expとして実装させる
5. **実行**: Colabで学習実行（並列可）
6. **記録**: EXP_SUMMARY.mdに結果を追記
7. **繰り返し**: 1日5-10実験を目標

## OOF分析でAIに頼むテンプレート

```
EXP001/output/child-exp003/oof.csv を分析して。
1. 全体のCV score
2. カテゴリ別（XX列）の誤差分布
3. 予測が大きく外れたサンプルTOP20の特徴
4. 前回のchild-exp002との比較（どこが改善/悪化したか）
結果をoutput/analysis/に保存して。
```

**分析結果を見て、次に何をやるかは人間が決める。AIにNext Actionを聞かない。**（本環境では＝分析サブエージェントの出力を鵜呑みにせず、**主セッションの Claude が人間モードで読んで次を決める**。user に Next Action を聞かない）

## コンペ選定基準

今後新しいコンペに参加する際：
- **反復速度**: 1実験の学習時間が短いか（＜30分が理想）
- **パイプラインの複雑さ**: シンプルなほどAIとの相性が良い
- **データサイズ**: 小さいほどColab無料枠で回せる
- **ドメイン**: 自分に知識があるか（野球関連は強み）

## 禁止事項

- **「精度上げろ」「改善して」のアバウトな指示**: 具体的な仮説を添える
- **1コンペ1 notebook全部入り**: EXP+child-exp構成を使う
- **結果を見ずに次のsubmit**: OOF分析してから次を決める
- **5コンペ以上の同時参加**: リソースが分散して全部中途半端になる
- **Kaggle kernelで開発サイクルを回す**: 反復が遅すぎる。開発はローカル+Colab
- **ローカルPCでML学習を実行する**: Celeron N4500 / 4GB RAMでは不可。Colab/RPi5を使う

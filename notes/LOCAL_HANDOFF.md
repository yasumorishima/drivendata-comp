# PC の Claude Code に貼る依頼文

PC で `drivendata-comp` を開き（無ければ `git clone https://github.com/yasumorishima/drivendata-comp`）、
`git pull` してから、Claude Code に下の枠の中をそのまま貼ってください。

---

```
DrivenData の作業を cloud の Claude Code セッションで続けられるように、
この PC にある DrivenData 関係のメモとデータを cloud から見える場所へ移してください。

1. 探す
   - この repo の外も含めて、DrivenData の開催中のコンペ（締切済みのものは不要）や、
     学習・提出の自動化の仕組みに関係するメモ・ノート・実験記録・Notebook・スクリーンショット・設定ファイル・データ・モデル・提出 zip を探す。
     候補：ホーム直下、Documents、Desktop、Downloads、~/.claude 配下のメモ、Colab の DL 物、Obsidian 等。
   - 見つけたものを「メモ類」と「データ類」に分け、パス・サイズの一覧を先に私に見せて確認を取る。

2. メモ類（自分で書いたもの）→ notes/ に入れて commit・push
   - notes/<コンペ名>/ にコピー（Markdown 以外はそのまま、長いものは要点の .md も添える）。
   - 公開 repo なので、パスワード・API キー・トークン・メールアドレス、配布データの中身は入れない（見つけたら伏せる）。

3. データ類（配布物・学習済みモデル・提出 zip など）→ 下書きリリースに上げる
   - bash tools/local_push.sh <name> <ファイルかフォルダ>
     （<name> は phonetic / word / models / submissions など。gh にログインしておく）
   - git には絶対に入れない。

4. notes/LOCAL_INVENTORY.md に「何を・どこから・どこへ（notes/ のパス or data-cache の <name>）」を表で書き、
   push する。
```

---

終わったら cloud のセッションに「PC から上げた」と伝えてください。cloud 側で `bash tools/cloud_setup.sh` を実行して取り込みます。

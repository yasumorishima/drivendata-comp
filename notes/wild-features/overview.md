# Where the Wild Features Are（2026-11-23 に見つけた・11-24 に参加）

- 場所：DrivenData ではなく **K-12 AI Infrastructure Platform**（`https://platform.k12-ai-infrastructure.org/competitions/4/literacy-speech-modeling/`）。運営は DrivenData。
  - アカウントは DrivenData と別。11-24 に PC の Claude が DrivenData と同じメール・パスワードで作ったので、ログインは既存の Secrets `DRIVENDATA_EMAIL`・`DRIVENDATA_PASSWORD` を使う（`K12_*` は無い）。フォームは `/accounts/login/` の `login`・`password`（django-allauth）。
  - ルールの要点は `rules.md`。
- 期間：2026-11-19 開始。Phase 1（予測の提出）は **2027-01-19** まで（登録・チームもここで締切）。Phase 2（上位 15 チームの write-up）は 2027-02-08 まで。
- 賞金 $50,000（1 位 $20,000・2 位 $15,000・3 位 $10,000・4 位 $5,000）。**順位でなく write-up の審査で決まる**（上位 15 位以内が条件）。上位 4 チームはコードの再現と MIT ライセンスでの公開が要る。
- 参加資格（公式ルール）：個人で、米国の法または居住地の法が参加・受賞を禁じない国の居住者、米国の制裁対象でない、18 歳以上。ロシア在住は対象外。**日本在住の日本人は参加でき、賞金も対象**。
- 外部データ：Open external data（CC0・CC-BY など商用可の公開データと学習済みモデルは可）。
- 課題：幼稚園児（902 人、24,518 本の音声、16 kHz の flac、匿名化・雑音除去済み）の年度初めの読みの検査（音をつなぐ・音を消す・文の復唱）の音声と採点から、年度末の読みの点数を予測する。
  - 1 日 3 本まで。テストの音声を人が聞くこと・手で書き起こすことは禁止。
  - write-up は「採点の正誤を超えて、子どもの声のどの特徴が効くか、どれだけ足すか」を、解釈できる形で示す。大人の声や雑音でなく子どもの声から来ていることも示す。
- 学習データ約 2 GB（train_audio.tar 1.2 GB・test_audio.tar 810 MB）。
- 手持ちとの相性：On Top of Pasketti（子どもの音声・音素）の仕組み（Kaggle の GPU・音声の前処理）が流用できる。

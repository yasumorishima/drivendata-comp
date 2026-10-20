# GEMS Prize（Geologic Enhanced Mapping System）— 調べたこと（2026-10-20）

出典：DrivenData のコンペページ（`/competitions/306/competition-doe-gems/`・problem description = page/967）、
公式ルール https://www.nlr.gov/docs/fy26osti/96647.pdf（19 ページ）、HeroX の案内。

## 🔴 参加資格（公式ルール 1.3）

> An individual prize competitor (who is not competing as a member of a group) must be a U.S. citizen or permanent resident.
> A group of individuals competing as one team may win, provided the designated team captain is a U.S. citizen or permanent resident.
> Individuals competing as part of a team are eligible to participate if they are legally authorized to work in the United States.

- 「To compete in the GEMS Prize, competitors must comply with the eligibility requirements below」＝**賞金だけでなく参加の条件**。
- 登録時にルールへの同意、賞金時に資格の宣誓（under penalty of perjury）がある。
- ⇒ user が米国市民・永住者でなければ参加できない。**user の確認が取れるまで登録（Compete!）しない。**

## 課題

- 西ネバダ（GeoDAWN 調査域）で、地表の断層の位置を 100 m 格子の確率ラスタで予測する。
- 入力：`training_features.tif`（EPSG 32611・100 m・多層：電気伝導度・標高の傾き・ひずみ速度・重力異常・磁気・地震密度など）、
  1 m DEM のリンク集（`1m_DEM_links.csv`）。ラベルは USGS 第四紀断層と INGENIOUS（ベクタ・ラスタ）。外部データ可（ライセンスがあり主催者と共有できるもの）。
- テスト：USGS に無い、専門家が新しく引いた断層。領域を塊に分けて public / private。
- 評価：距離重み付き Tversky 指数（α=0.2 で偽陽性に甘く、β=0.8 で見落としに厳しい）。三角カーネル R=300 m（3 画素）。
- 提出：同じ範囲・同じ解像度の float32 1 層 GeoTIFF（0〜1）。
- 賞：Initial（private の上位 5 に各 $10K）と Final（専門家が全提出を見て増やしたラベルで再採点、上位 5 に計 $250K）。提出は 1 本選ぶ。
- 締切 2026-12-03 23:59 UTC。参考解：https://github.com/drivendataorg/gems-prize-reference-solution
- 公開 LB（10-20）：1 位 0.4413・5 位 0.3539・7 位 0.3416（提出 12〜43 本）。
- 1 日の提出上限：未確認（提出ページは登録後に見える）。

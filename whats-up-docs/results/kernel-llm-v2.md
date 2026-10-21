# llm-v2 (2026-10-21T02:49Z, commit 53874e4, status )

memo: llm-v2: Qwen2.5-7B ゼロショット要旨を学習 900 本にも作る（llm-v1 と同じ設定）。文との 2 語並び一致を LightGBM の特徴として学ばせるため

```
extract: ROUGE-2 F1 0.1812 (median 202 words)
llm_ext: ROUGE-2 F1 0.1681 (median 234 words)
ext_llm: ROUGE-2 F1 0.1643 (median 358 words)
llm: ROUGE-2 F1 0.1474 (median 154 words)
```

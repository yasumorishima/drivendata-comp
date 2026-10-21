# lora-v1 (2026-10-21T02:36Z, commit 4c77259, status complete)

memo: LoRA v1: 10-21 01:09 に push 済みの kernel v2 の完了を待って出力を集める（前回は PRE_LRT の取り込みで落ちた）

```
extract: ROUGE-2 F1 0.1812 (median 202 words)
lora_ext: ROUGE-2 F1 0.1437 (median 193 words)
lora: ROUGE-2 F1 0.1092 (median 133 words)
lora beats extract on 10/100 val papers
model Qwen/Qwen2.5-1.5B-Instruct epochs 2 lr 0.0002
```

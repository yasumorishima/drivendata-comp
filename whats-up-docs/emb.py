"""Meaning-level match of each sentence (MiniLM sentence embeddings, CPU), as features for LightGBM:
cosine to the LLM abstract (v4's guide, but beyond shared word pairs), to the paper's opening and to the
paper's mean sentence.

  python whats-up-docs/emb.py <data_dir> <shard> <n_shards> <out.csv>
Output rows: paper_id, pos (sentence position in the paper's table), e_llm, e_open, e_cent.
Runs on LLM on CPU as a "script" (the material argument is the data dir), or locally with shard 0 of 1.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractive import table  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Apache-2.0
LLM = os.environ.get("EMB_LLM", "kout-llm-cpu-v1")


class Enc:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.mod = AutoModel.from_pretrained(MODEL).eval()

    def __call__(self, texts, max_len=96, bs=128):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 384), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                idx = order[i:i + bs]
                b = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                             return_tensors="pt")
                h = self.mod(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out[idx] = torch.nn.functional.normalize(e, dim=-1).numpy()
        return out


if __name__ == "__main__":
    d, shard, n, out = sys.argv[1].rstrip("/") + "/", int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    torch.set_num_threads(os.cpu_count())
    frames = [pd.read_csv(d + "train.csv"), pd.read_csv(d + "test_features.csv")]
    df = pd.concat([f[["paper_id", "text"]] for f in frames], ignore_index=True).iloc[shard::n]
    llm = pd.read_csv(d + f"../{LLM}/preds.csv").fillna("").set_index("paper_id").llm
    enc, rows, t0 = Enc(), [], time.time()
    for k, (pid, text) in enumerate(zip(df.paper_id, df.text)):
        t = table([text])
        if t.empty:
            continue
        s = enc(t.sent.tolist())
        g = enc([llm.get(pid, "") or " ", " ".join(str(text).split()[:250])], max_len=256)
        c = s.mean(0)
        c /= max(np.linalg.norm(c), 1e-6)
        rows.append(pd.DataFrame({"paper_id": pid, "pos": np.arange(len(t)), "e_llm": s @ g[0],
                                  "e_open": s @ g[1], "e_cent": s @ c}))
        if k % 20 == 0:
            print(f"{k + 1}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    pd.concat(rows).round(4).to_csv(out, index=False)
    print("done", len(df), f"{time.time() - t0:.0f}s")

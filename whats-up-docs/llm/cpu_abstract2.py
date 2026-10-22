"""Second zero-shot abstract (Qwen2.5-7B 4-bit, llama.cpp, CPU runner), with a different prompt from
cpu_generate.py: the LLM sees the paper's top-30 v4 sentences (select_prep.py) and builds the abstract
by copying and lightly trimming them, so its word pairs point at sentences more directly.

  python whats-up-docs/llm/cpu_abstract2.py <cands.csv> <shard> <n_shards> <out.csv> [skip_preds.csv]
Output: paper_id, split, llm (the abstract).
"""

import json
import os
import re
import sys
import time

import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

PROMPT = """Below are the opening of a social science research paper and {n} numbered sentences taken from it.

Write the paper's abstract (about 180 words, one paragraph, plain text) the way its authors would: first the \
research question and why it matters, then the data and method, then the main findings, then the implications. \
Build it from the sentences below: copy their wording, shortening or joining them where needed, rather than \
paraphrasing. Do not add a title, headings, citations, numbers of sentences or anything that is not in the paper.

OPENING OF THE PAPER:
{opening}

SENTENCES:
{sents}

ABSTRACT:"""

REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"  # Apache-2.0
FILES = ["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"]
SYSTEM = "You are an expert social scientist who writes precise academic abstracts."

if __name__ == "__main__":
    _, path, shard, n, out = sys.argv[:5]
    df = pd.read_csv(path).fillna("")
    if len(sys.argv) > 5 and os.path.exists(sys.argv[5]):
        done = pd.read_csv(sys.argv[5]).fillna("")
        df = df[~df.paper_id.isin(done.paper_id[done.llm.astype(str).str.len() > 0])]
    df = df.iloc[int(shard)::int(n)]
    paths = [hf_hub_download(REPO, f, local_dir=os.path.expanduser("~/gguf")) for f in FILES]
    llm = Llama(paths[0], n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    rows, t0 = [], time.time()
    for k, r in enumerate(df.itertuples()):
        cands, text = json.loads(r.cands), ""
        for _ in range(4):
            sents = "\n".join(f"[{j + 1}] {s}" for j, (_, s) in enumerate(cands))
            try:
                o = llm.create_chat_completion(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT.format(n=len(cands), opening=r.opening, sents=sents)}],
                    temperature=0, max_tokens=330, repeat_penalty=1.05)
                text = o["choices"][0]["message"]["content"]
                break
            except ValueError as e:
                print("shorten:", e, flush=True)
                cands = [[p, " ".join(s.split()[: int(len(s.split()) * 0.7)])] for p, s in cands]
        text = re.sub(r"^\s*(abstract|summary)\s*[:.]\s*", "", text, flags=re.I)
        text = re.sub(r"\[\d+\]", "", text).replace("\n", " ").strip()
        rows.append({"paper_id": r.paper_id, "split": r.split, "llm": text})
        print(f"{k + 1}/{len(df)} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

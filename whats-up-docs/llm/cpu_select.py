"""Zero-shot sentence selection with Qwen2.5-7B-Instruct (4-bit GGUF, llama.cpp) on a CPU runner:
the LLM reads the paper's top-30 candidate sentences (select_prep.py) and names the ones the authors'
abstract would be built from (the 2nd-place public solution does this with a paid LLM).

  python whats-up-docs/llm/cpu_select.py <cands.csv> <shard> <n_shards> <out.csv> [skip_preds.csv]
Output: paper_id, split, llm (positions in the paper's sentence table, most important first, "12 3 40").
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

The authors' abstract states the paper's research question, its data or method, its main findings and their \
implications. Choose the 6 to 10 sentences whose content and wording the abstract would most likely contain. \
Prefer sentences that summarize the whole paper over details, examples, citations or statistics tables.

OPENING OF THE PAPER:
{opening}

SENTENCES:
{sents}

Answer with the numbers of the chosen sentences only, most important first, separated by commas."""

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
        cands = json.loads(r.cands)
        text, sel = "", []
        for _ in range(4):
            sents = "\n".join(f"[{j + 1}] {s}" for j, (_, s) in enumerate(cands))
            try:
                o = llm.create_chat_completion(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT.format(n=len(cands), opening=r.opening, sents=sents)}],
                    temperature=0, max_tokens=60)
                text = o["choices"][0]["message"]["content"]
                break
            except ValueError as e:  # too long: cut every candidate shorter
                print("shorten:", e, flush=True)
                cands = [[p, " ".join(s.split()[: int(len(s.split()) * 0.7)])] for p, s in cands]
        for x in re.findall(r"\d+", text):
            j = int(x) - 1
            if 0 <= j < len(cands) and cands[j][0] not in sel:
                sel.append(cands[j][0])
        rows.append({"paper_id": r.paper_id, "split": r.split, "llm": " ".join(map(str, sel)), "raw": text[:200]})
        print(f"{k + 1}/{len(df)} {time.time() - t0:.0f}s picked {len(sel)}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

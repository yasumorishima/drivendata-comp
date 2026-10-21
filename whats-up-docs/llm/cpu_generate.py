"""Zero-shot abstracts with Qwen2.5-7B-Instruct (4-bit GGUF, llama.cpp) on a CPU runner.

  python whats-up-docs/llm/cpu_generate.py <material.csv> <shard> <n_shards> <out.csv>
Same prompt as summarize_llm.py (llm-v1). One paper takes ~140 s on 4 cores, so the 1,345 papers
are split over 20 GitHub Actions jobs (.github/workflows/llm-cpu.yml).
"""

import os
import re
import sys
import time

import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# same text as summarize_llm.py (that file imports torch, which the CPU runner does not have)
PROMPT = """Below are the opening of a social science research paper and the sentences that best describe its contribution.

Write the paper's abstract: one paragraph of about 170 words, in plain text, as the authors would write it \
(for example "This article examines ...", "Using data from ..., we find ..."). State the question, the data or \
method, the main findings and the implications. Reuse the paper's own key terms and phrases wherever possible. \
Do not add a title, headings, citations or anything that is not in the paper.

OPENING OF THE PAPER:
{opening}

KEY SENTENCES:
{material}

ABSTRACT:"""

REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"  # Apache-2.0
FILES = ["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"]
SYSTEM = "You are an expert social scientist who writes precise academic abstracts."

if __name__ == "__main__":
    _, path, shard, n, out = sys.argv
    df = pd.read_csv(path).fillna("")
    df = df.iloc[int(shard)::int(n)]
    paths = [hf_hub_download(REPO, f, local_dir=os.path.expanduser("~/gguf")) for f in FILES]
    llm = Llama(paths[0], n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
    rows, t0 = [], time.time()
    for k, r in enumerate(df.itertuples()):
        o = llm.create_chat_completion(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": PROMPT.format(opening=r.opening, material=r.material)}],
            temperature=0, max_tokens=330, repeat_penalty=1.05)
        text = o["choices"][0]["message"]["content"]
        text = re.sub(r"^\s*(abstract|summary)\s*[:.]\s*", "", text, flags=re.I).replace("\n", " ").strip()
        rows.append({"paper_id": r.paper_id, "split": r.split, "llm": text})
        print(f"{k + 1}/{len(df)} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)  # keep partial output if the job times out

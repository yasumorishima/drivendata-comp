"""Input for cpu_abstract4.py (fourth abstract, llm-abs4-v1): the opening and the paper's contribution statements.

llm-v1 / llm-abs2-v1 read the top model sentences, llm-abs3-v1 the concluding part. This one reads the
sentences anywhere in the paper where the authors say what they do or find ("this paper", "we argue",
"our results" ...), in document order, up to 650 words (topped up with the introduction after the opening when
there are fewer than 300 words of them).

  python whats-up-docs/llm/abs4_prep.py <data_dir> <out.csv>
Columns: paper_id, split, opening, material.
"""

import re
import sys

import pandas as pd

OPEN, END = 150, 650
CUE = re.compile(r"\b(this (paper|article|study|chapter)|in this (paper|article|study)|we (argue|find|show|examine|"
                 r"investigate|propose|demonstrate|suggest|use|test|develop|analy[sz]e|explore|contribute)|"
                 r"our (results|findings|analysis|study|paper|argument|contribution)|the (results|findings) "
                 r"(show|suggest|indicate))\b", re.I)


def statements(text):
    body = re.sub(r"^#+ .*$", " ", str(text), flags=re.M)
    sents = re.split(r"(?<=[.!?])\s+", " ".join(body.split()))
    out, n = [], 0
    for s in sents[1:]:
        if CUE.search(s) and 5 <= len(s.split()) <= 80:
            out.append(s)
            n += len(s.split())
            if n >= END:
                break
    if n < 300:  # few statements: add the introduction that follows the opening
        out.append(" ".join(" ".join(body.split()).split()[OPEN:OPEN + END - n]))
    return " ".join(out)


if __name__ == "__main__":
    d, out = sys.argv[1].rstrip("/") + "/", sys.argv[2]
    rows = []
    for split, f in [("test", "test_features.csv"), ("train", "train.csv")]:
        df = pd.read_csv(d + f)
        for p, t in zip(df.paper_id, df.text):
            rows.append({"paper_id": p, "split": split, "opening": " ".join(str(t).split()[:OPEN]), "material": statements(t)})
    pd.DataFrame(rows).to_csv(out, index=False)
    print("rows", len(rows))

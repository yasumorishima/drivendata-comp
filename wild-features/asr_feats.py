"""ASR features for sentence_repetition clips (faster-whisper base.en, MIT weights; CPU int8).

Each clip is transcribed on its own. Per clip: recall of the expected words, share of extra words, word count,
mean log-probability and no-speech probability from the decoder. Transcripts stay outside the repo.
  python wild-features/asr_feats.py <audio_dir> <clip_metadata.csv> <out.parquet>
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel


def words(x):
    return re.findall(r"[a-z']+", str(x).lower())


def main():
    adir, meta, out = sys.argv[1:4]
    c = pd.read_csv(meta)
    c = c[c.task == "sentence_repetition"]
    m = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
    rows = []
    for k, (f, e) in enumerate(zip(c.filename, c.expected_text)):
        r = {"filename": f}
        try:
            y, _ = sf.read(os.path.join(adir, os.path.basename(f)), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            segs = list(m.transcribe(y, beam_size=1, language="en", condition_on_previous_text=False,
                                     without_timestamps=True)[0])
            h, ew = words(" ".join(s.text for s in segs)), words(e)
            hs, es = set(h), set(ew)
            r.update(s_recall=len(hs & es) / max(len(es), 1), s_extra=len(hs - es) / max(len(hs), 1),
                     s_nwords=len(h) / max(len(ew), 1),
                     s_logp=float(np.mean([s.avg_logprob for s in segs])) if segs else np.nan,
                     s_nospeech=float(np.mean([s.no_speech_prob for s in segs])) if segs else np.nan,
                     s_text=" ".join(h))
        except Exception:
            pass
        rows.append(r)
        if k % 1000 == 0:
            print("done", k, flush=True)
            pd.DataFrame(rows).to_parquet(out)
    pd.DataFrame(rows).to_parquet(out)
    print("clips", len(rows))


if __name__ == "__main__":
    main()

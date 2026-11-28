"""Per-clip acoustic features for Wild Features (interpretable, CPU only, no pretrained model).

Each clip is processed on its own (no statistics from other clips or from the test set).
  python wild-features/audio_feats.py <audio_root> <clip_metadata.csv> <out.parquet> [n_proc] [limit]
audio_root holds train_audio/ and test_audio/. The output stays outside the repo (~/drivendata-env/).
Columns: voiced time and share, number of voiced segments, time to the first voiced segment (response latency),
longest pause, trailing silence, loudness, pitch (yin, median and spread over voiced frames), spectral centroid,
zero-crossing rate, 13 MFCC means.
"""
import os
import sys
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
HOP = 160  # 10 ms


def one(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    n = len(y) / SR
    out = {"a_len": n}
    if n < 0.05 or not np.any(y):
        return out
    seg = librosa.effects.split(y, top_db=30, frame_length=400, hop_length=HOP) / SR
    seg = seg[(seg[:, 1] - seg[:, 0]) >= 0.05] if len(seg) else seg
    if len(seg):
        v = float((seg[:, 1] - seg[:, 0]).sum())
        gaps = seg[1:, 0] - seg[:-1, 1] if len(seg) > 1 else np.array([0.0])
        out.update(a_voiced=v, a_vratio=v / n, a_nseg=len(seg), a_onset=float(seg[0, 0]),
                   a_trail=float(n - seg[-1, 1]), a_maxpause=float(gaps.max()), a_span=float(seg[-1, 1] - seg[0, 0]))
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-6)
    out.update(a_db_p90=float(np.percentile(db, 90)), a_db_range=float(np.percentile(db, 90) - np.percentile(db, 10)))
    loud = db > np.percentile(db, 90) - 25
    f0 = librosa.yin(y, fmin=80, fmax=600, sr=SR, frame_length=1024, hop_length=HOP)
    m = min(len(f0), len(loud))
    fv = f0[:m][loud[:m]]
    if len(fv) > 5:
        out.update(a_f0_med=float(np.median(fv)), a_f0_iqr=float(np.subtract(*np.percentile(fv, [75, 25]))))
    out["a_cent"] = float(librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0][loud[:m] if m else slice(None)].mean()) if m else np.nan
    out["a_zcr"] = float(librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=HOP)[0].mean())
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=HOP)
    for k in range(13):
        out[f"a_mfcc{k}"] = float(mf[k].mean())
    return out


def work(args):
    fn, path = args
    try:
        r = one(path)
    except Exception:
        r = {}
    r["filename"] = fn
    return r


def main():
    root, meta, out = sys.argv[1], sys.argv[2], sys.argv[3]
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    c = pd.read_csv(meta)
    if len(sys.argv) > 5:
        c = c.head(int(sys.argv[5]))
    part = "train_audio" if "train" in os.path.basename(meta) else "test_audio"
    jobs = [(f, os.path.join(root, part, os.path.basename(f))) for f in c.filename]
    with Pool(nproc) as p:
        rows = p.map(work, jobs, chunksize=32)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print("clips", len(df), "failed", int(df.a_len.isna().sum()) if "a_len" in df else len(df))


if __name__ == "__main__":
    main()

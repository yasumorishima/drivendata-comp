# requests/

cloud セッションから GitHub Actions を動かすための依頼ファイル（cloud は dispatch できないため）。

| ファイル | 動くワークフロー | 書式 |
|---|---|---|
| `dispatch*.json` | `Cloud Dispatch` → 指定のワークフロー | `.github/workflows/cloud-dispatch.yml` 先頭 |
| `data-cache.json` | `Data Cache` | `.github/workflows/data-cache.yml` 先頭 |
| `selftest-data.json` | `Data Cache Selftest` | `.github/workflows/data-cache-selftest.yml` 先頭 |

同じ中身の再 push は動かない。毎回 `memo` を変える。

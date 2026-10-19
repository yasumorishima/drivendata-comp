#!/bin/bash
# cloud セッションで DrivenData のデータを使えるようにする。
#   usage: bash tools/cloud_setup.sh [name ...]     (省略時: 下書きリリースにある全部)
# 下書きリリース "data-cache" の添付 "<name>__<file>" を $DD_ENV/data/<name>/<file> に取る。
# 分割（<file>.partNN）は結合し、<name>__SHA256SUMS.txt で確認する。取得済みは飛ばす。
# 添付を置くのは .github/workflows/data-cache.yml（DrivenData から）か
# tools/local_push.sh（ユーザーの PC から）。
# $DD_ENV の中身は repo に入れない（DrivenData の規約で再配布禁止）。
set -euo pipefail

REPO="${DD_REPO:-yasumorishima/drivendata-comp}"
ENVD="${DD_ENV:-$HOME/drivendata-env}"
TAG=data-cache
mkdir -p "$ENVD/data" "$ENVD/dl"

rel=$(gh api "repos/$REPO/releases?per_page=100")
id=$(printf '%s' "$rel" | python3 -c 'import json,sys; r=[x for x in json.load(sys.stdin) if x.get("draft") and x.get("tag_name")=="data-cache"]; print(r[0]["id"] if r else "")')
[ -n "$id" ] || { echo "no draft release data-cache: push requests/data-cache.json or run tools/local_push.sh on the PC"; exit 1; }
assets=$(gh api "repos/$REPO/releases/$id/assets?per_page=100" --paginate)

want=("$@")
names=$(printf '%s' "$assets" | python3 -c '
import json,sys
want=set(sys.argv[1:])
for a in json.load(sys.stdin):
    n=a["name"]
    if "__" in n and (not want or n.split("__",1)[0] in want):
        print(a["id"], a["size"], n)' "${want[@]}")

fetch() {  # asset-id out-file   (curl: release files are served from another host)
  curl -fsSL --retry 3 --max-time 3600 -H "Accept: application/octet-stream" \
       "https://api.github.com/repos/$REPO/releases/assets/$1" -o "$2"
}

while read -r aid size n; do
  [ -n "${aid:-}" ] || continue
  out="$ENVD/dl/$n"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then continue; fi
  echo "downloading $n ($((size/1024/1024)) MB)"
  fetch "$aid" "$out"
done <<< "$names"

# join parts, check hashes, place under data/<name>/
for sums in "$ENVD"/dl/*__SHA256SUMS.txt; do
  [ -f "$sums" ] || continue
  comp=$(basename "$sums" | sed 's/__SHA256SUMS.txt$//')
  mkdir -p "$ENVD/data/$comp"
  while read -r h base; do
    f="${base#${comp}__}"
    dst="$ENVD/data/$comp/$f"
    if [ -f "$dst" ] && [ -f "$dst.ok" ]; then continue; fi
    if ls "$ENVD/dl/$base".part* > /dev/null 2>&1; then
      cat "$ENVD/dl/$base".part* > "$dst"
    elif [ -f "$ENVD/dl/$base" ]; then
      cp "$ENVD/dl/$base" "$dst"
    else
      echo "missing $base"; continue
    fi
    echo "$h  $dst" | sha256sum -c --quiet - && touch "$dst.ok" && echo "ok $comp/$f"
  done < "$sums"
done

echo "data under: $ENVD/data"
du -sh "$ENVD/data"/* 2>/dev/null || true
echo "（展開が必要な zip はここで unzip する。展開先も repo の外に置く）"

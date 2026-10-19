#!/bin/bash
# ユーザーの PC で実行する：手元のデータを cloud セッションへ渡す。
#   usage: bash tools/local_push.sh <name> <file-or-dir> [...]
#     例: bash tools/local_push.sh phonetic ~/drivendata/data/phonetic
# 下書きリリース "data-cache"（push 権限のある人にしか見えない）に "<name>__<file>" で上げる。
# 2GB 超は分割、<name>__SHA256SUMS.txt も上げる（cloud 側は tools/cloud_setup.sh で取る）。
# 必要: gh（ログイン済み）、python3、sha256sum か shasum。
# メモ・ノート（自分で書いたもの）はこれではなく notes/ に置いて commit する。
set -euo pipefail

[ $# -ge 2 ] || { sed -n 2,9p "$0"; exit 1; }
name="$1"; shift
REPO="${DD_REPO:-yasumorishima/drivendata-comp}"
TAG=data-cache

sha() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

if ! gh release view "$TAG" -R "$REPO" > /dev/null 2>&1; then
  gh release create "$TAG" -R "$REPO" --draft --title data-cache \
    --notes "DrivenData 配布データの受け渡し用（下書き・非公開）。公開しないこと。"
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
sums="$tmp/${name}__SHA256SUMS.txt"
# keep entries already on the release so several runs for the same name add up
gh release download "$TAG" -R "$REPO" -p "${name}__SHA256SUMS.txt" -D "$tmp" 2> /dev/null || : > "$sums"

files=()
for p in "$@"; do
  if [ -d "$p" ]; then while IFS= read -r f; do files+=("$f"); done < <(find "$p" -maxdepth 1 -type f ! -name '.*' | sort)
  else files+=("$p"); fi
done

for f in "${files[@]}"; do
  base="${name}__$(basename "$f")"
  echo "== $base"
  grep -v "  $base\$" "$sums" > "$sums.new" || true; mv "$sums.new" "$sums"
  echo "$(sha "$f")  $base" >> "$sums"
  size=$(wc -c < "$f" | tr -d ' ')
  if [ "$size" -gt 1990000000 ]; then
    # split one part at a time (same names as GNU split -d -a 2: .part00, .part01, ...)
    i=0
    while :; do
      part="$tmp/$base.part$(printf %02d $i)"
      python3 -c 'import sys; f=open(sys.argv[1],"rb"); f.seek(int(sys.argv[2])*1900*1024*1024); d=f.read(1900*1024*1024); open(sys.argv[3],"wb").write(d); sys.exit(0 if d else 1)' "$f" "$i" "$part" || { rm -f "$part"; break; }
      gh release upload "$TAG" "$part" -R "$REPO" --clobber; rm -f "$part"; i=$((i+1))
    done
  else
    # the asset name is the file name, so upload through a link named "<name>__<file>"
    ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$tmp/$base" 2> /dev/null || cp "$f" "$tmp/$base"
    gh release upload "$TAG" "$tmp/$base" -R "$REPO" --clobber; rm -f "$tmp/$base"
  fi
done
gh release upload "$TAG" "$sums" -R "$REPO" --clobber
echo "done. cloud 側で: bash tools/cloud_setup.sh $name"

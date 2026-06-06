#!/usr/bin/env bash
# Resize the original 1024x1024 effect images into two button-thumbnail sizes:
#   - monitor:    240px (large kiosk monitor, side-column buttons)
#   - smartphone: 120px (phone buttons; small so image + label fit on one row)
# Output goes to images/<size>/ AND is copied into ui/public/effects/<size>/
# so Vite serves them at /effects/<size>/<name>.jpeg.
set -euo pipefail

cd "$(dirname "$0")"
SRC="original"
UI_PUBLIC="../ui/public/effects"

resize_into() {
  size="$1"
  px="$2"
  mkdir -p "$size" "$UI_PUBLIC/$size"
  for f in "$SRC"/*.jpeg; do
    name="$(basename "$f")"
    sips -s format jpeg -z "$px" "$px" "$f" --out "$size/$name" >/dev/null
    cp "$size/$name" "$UI_PUBLIC/$size/$name"
  done
  echo "Resized to ${px}px -> $size/ and $UI_PUBLIC/$size/"
}

resize_into monitor 240
resize_into smartphone 120

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PAPER_DIR"

pandoc PAPER.md \
    -o PAPER.pdf \
    --pdf-engine=xelatex \
    --resource-path=. \
    -H scripts/preamble.tex \
    -V mainfont="Palatino" \
    -V mathfont="Palatino" \
    -V monofont="Menlo" \
    -V fontsize=11pt \
    -V geometry:"margin=1in" \
    -V linestretch=1.15

echo "$PAPER_DIR/PAPER.pdf"

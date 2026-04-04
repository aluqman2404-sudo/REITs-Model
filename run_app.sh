#!/bin/bash
# UK Housing Model — Streamlit launcher
set -e
cd "$(dirname "$0")"

# Add user Python bin to PATH (covers pip install --user installs)
export PATH="$HOME/Library/Python/3.12/bin:$PATH"

echo "Starting UK Housing Model dashboard..."
echo "Open http://localhost:8501 in your browser"
echo "(Press Ctrl+C to stop)"
echo ""

streamlit run src/ui/app.py \
  --server.port 8501 \
  --server.headless false \
  --theme.base light \
  --theme.primaryColor "#12355b" \
  --theme.backgroundColor "#fbfaf7" \
  --theme.secondaryBackgroundColor "#f0ebe1" \
  --theme.textColor "#132238"

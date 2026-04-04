#!/usr/bin/env bash
# Stage 1: Environment setup script
# Run once from the housing_model/ directory:
#   chmod +x setup.sh && ./setup.sh

set -e

echo "==> Creating virtual environment..."
python3 -m venv .venv

echo "==> Activating virtual environment..."
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Copying .env template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "    .env created — fill in your API keys before running ingestion."
else
    echo "    .env already exists, skipping."
fi

echo ""
echo "Setup complete. To activate the environment in future sessions:"
echo "    source .venv/bin/activate"
echo ""
echo "To launch the Streamlit app (Stage 7):"
echo "    ./run_app.sh"
echo "    # or: streamlit run src/ui/app.py"

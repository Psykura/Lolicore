curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Psykura/Lolicore
cd Lolicore
uv sync
uv run train.py
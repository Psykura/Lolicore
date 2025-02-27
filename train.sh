curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/Psykura/Lolicore
cd Lolicore
uv sync
uv run train.py
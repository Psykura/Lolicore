# Lolicore: Simple, Playable and Flexible Language Model

Lolicore is a experimental language model written with jax and flax.

You can train a mixture of expert model that support sharding, parallel on both TPU and GPU devices by running the following lines of code:

```bash
# clone repo
git clone https://github.com/Psykura/Lolicore
cd Lolicore
# install deps
uv sync
# start training
uv run train.py
```

```bash
curl -LsSf https://raw.githubusercontent.com/Psykura/Lolicore/refs/heads/main/train.sh | sh
```
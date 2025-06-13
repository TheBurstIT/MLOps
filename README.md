# Crypto LOB Micro Move Classifier

This repository shows a minimal setup for limit order book classification using
simple utilities. Configuration files in `configs/` follow the Hydra layout so
parameters can be overridden from the command line.
Data retrieval and training are described as DVC stages.

## Usage

```bash
# download a tiny sample dataset
python -m crypto_lob_micro_move.cli download-data --output_dir data/raw

# run training with default settings
python -m crypto_lob_micro_move.cli train

# override parameters
python -m crypto_lob_micro_move.cli train epochs=2 data.path=data/raw

# export trained checkpoint
python -m crypto_lob_micro_move.export.onnx --ckpt model.json --out model.onnx
```

The downloader stores a small `sample.csv` so the tests and example pipeline can
run without external data.

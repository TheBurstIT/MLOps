# Crypto LOB Micro Move Classifier

This repository contains a tiny example of a limit order book classifier. A
small dataset can be generated locally and a lightweight training loop is
provided. Configuration files under `configs/` follow a Hydra-like layout so
parameters may be overridden on the command line. The `dvc.yaml` file defines
stages for dataset download and training.

## Usage

```bash
# download a tiny sample dataset
PYTHONPATH=./src python -m crypto_lob_micro_move.cli download-data --output_dir data/raw

# run training with default settings
PYTHONPATH=./src python -m crypto_lob_micro_move.cli train

# override parameters
PYTHONPATH=./src python -m crypto_lob_micro_move.cli train epochs=2 data.path=data/raw

# export trained checkpoint
PYTHONPATH=./src python -m crypto_lob_micro_move.cli export --ckpt model.json --out model.onnx
```

The downloader stores a small `sample.csv` so the tests and example pipeline can
run without external data.

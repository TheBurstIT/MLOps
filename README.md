# Crypto LOB Micro Move Classifier

This repository shows a minimal setup for limit order book classification with
DVC, Hydra and MLflow logging.

## Usage

```bash
# download sample dataset (requires kagglehub credentials for full data)
python -m crypto_lob_micro_move.cli download-data --output_dir data/raw

# run training with default settings
python -m crypto_lob_micro_move.cli train --data.path=data/raw --epochs=1

# export trained checkpoint to ONNX
python -m crypto_lob_micro_move.export.onnx --ckpt model.ckpt --out model.onnx
```

The downloader stores a small `sample.csv` so the tests and example pipeline can
run without external data.

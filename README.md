# Crypto LOB Micro Move Classifier

This repository provides a toy example of a limit order book classifier.
It demonstrates dataset management, a simple PyTorch Lightning model and
Hydra-based configuration.

## Usage

```bash
# download sample dataset (requires kagglehub credentials for full data)
python -m crypto_lob_micro_move.cli download-data --output_dir data/raw

# run training with default settings
python -m crypto_lob_micro_move.cli train --data.path=data/raw --train.epochs=1

# export trained checkpoint to ONNX
python -m crypto_lob_micro_move.export.onnx --ckpt model.ckpt --out model.onnx
```

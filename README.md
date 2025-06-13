# Crypto LOB Micro Move Classifier

This repository provides a small project for limit order book
classification. It includes a dataset downloader and a stub training
routine.

## Usage

```bash
# download dataset (requires kagglehub credentials)
python -m crypto_lob_micro_move.cli download-data --output_dir data/raw

# run a simple training loop
python -m crypto_lob_micro_move.cli train --data_dir data/raw --epochs 1
```

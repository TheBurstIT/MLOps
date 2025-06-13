# Crypto LOB Micro Move Classifier

## Overview

A minimal example project for classifying limit order book (LOB) micro moves.
The repository includes a dataset downloader, a stub training routine and a few
utilities for preparing models for production.

## Setup

Install dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

This project uses [DVC](https://dvc.org) to manage large files. After cloning run:

```bash
dvc init
dvc remote add -d storage s3://example-bucket/dvc-storage
```

If you use [MLflow](https://mlflow.org) for experiment tracking set your tracking
URI (optional):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Train

Download the dataset (requires `kagglehub` credentials) and run the training loop:

```bash
python -m crypto_lob_micro_move.cli download-data --output_dir data/raw
python -m crypto_lob_micro_move.cli train --data_dir data/raw --epochs 1
```

Data is expected under `data/raw`.

## Production preparation

Convert the trained ONNX model to TensorRT before serving:

```bash
python -m crypto_lob_micro_move.cli onnx2trt path/to/model.onnx docker/triton/models/micro_move/1/model.plan
```

## Infer

Start Triton using the provided model repository:

```bash
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v $(pwd)/docker/triton/models:/models \
    nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models
```

Send a sample request:

```bash
curl -X POST -H 'Content-Type: application/json' \
    -d '{"inputs":[{"name":"input","shape":[1],"datatype":"FP32","data":[0.0]}]}' \
    http://localhost:8000/v2/models/micro_move/infer
```

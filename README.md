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

## DVC Usage

This project uses [DVC](https://dvc.org) to version large datasets. After cloning the repository run:

```bash
dvc init
dvc remote add -d storage s3://example-bucket/dvc-storage
```

To download data that was previously pushed to the remote storage:

```bash
dvc pull
```

After adding or modifying data tracked by DVC, upload it with:

```bash
dvc push
```

## Triton Inference Server

Convert your ONNX model to TensorRT before serving:

```bash
python -m crypto_lob_micro_move.cli onnx2trt path/to/model.onnx path/to/model.plan
```

Run Triton using the provided model repository:

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


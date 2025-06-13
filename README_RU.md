# Классификатор микродвижений в криптовалютном стакане

## Обзор

Минимальный пример проекта для классификации микродвижений лимитного ордера (LOB).
Репозиторий включает загрузчик датасета, заглушку тренировочного цикла и несколько
утилит для подготовки моделей к продакшену.

## Установка

Установите зависимости с помощью [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Если окружение CI не имеет доступа к интернету, соберите Docker-образ
с уже установленными зависимостями:

```bash
docker build -t micro-move-dev -f docker/Dockerfile .
```

Проект использует [DVC](https://dvc.org) для управления большими файлами. После клонирования выполните:

```bash
dvc init
dvc remote add -d storage s3://example-bucket/dvc-storage
```

Если вы используете [MLflow](https://mlflow.org) для отслеживания экспериментов, укажите URI трекинга (опционально):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Обучение

Скачайте датасет (требуются учётные данные `kagglehub`) и запустите тренировочный цикл:

```bash
python -m crypto_lob_micro_move.cli download-data --output_dir data/raw
python -m crypto_lob_micro_move.cli train --data_dir data/raw --dataset btc --epochs 1
```

Данные должны располагаться в каталоге `data/raw/<dataset>`, где `<dataset>` может принимать значения `btc`, `eth` или `ada`.

## Подготовка к продакшену

Перед развёртыванием конвертируйте обученную модель ONNX в TensorRT:

```bash
python -m crypto_lob_micro_move.cli onnx2trt path/to/model.onnx docker/triton/models/micro_move/1/model.plan
```

## Инференс

Запустите Triton с предоставленным репозиторием моделей:

```bash
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v $(pwd)/docker/triton/models:/models \
    nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models
```

Отправьте пример запроса:

```bash
curl -X POST -H 'Content-Type: application/json' \
    -d '{"inputs":[{"name":"input","shape":[1],"datatype":"FP32","data":[0.0]}]}' \
    http://localhost:8000/v2/models/micro_move/infer
```

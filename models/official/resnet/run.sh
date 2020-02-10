#!bin/bash

python resnet_main.py \
  --tpu=iacomagcp \
  --mode=train_and_eval \
  --data_dir=gs://iacoma-storage/tf_records \
  --model_dir=gs://iacoma-storage/ckpt-resnet101 \
  --train_batch_size=1024 \
  --train_steps=112590 \
  --iterations_per_loop=1251 \
  --use_tpu=true    \
  --precision=float32   2>&1 | tee tpu-resnet101.log

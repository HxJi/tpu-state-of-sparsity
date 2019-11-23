#!bin/bash

python resnet_main.py \
  --tpu=iacomagcp \
  --mode=train_and_eval \
  --data_dir=gs://iacoma-storage/tf_records \
  --model_dir=gs://iacoma-storage/ckpt-1 \
  --train_batch_size=1024 \
  --train_steps=112590 \
  --iterations_per_loop=1251 \
  --use_tpu=true    \
  --precision=float32   2>&1 | tee tpu.log

python resnet_main.py --mode=eval --eval_batch_size=64 --data_dir=/scratch/hj14/tf_record_latest --model_dir=/scratch/hj14/tpu_checkpoints/ckpt-2 
--use_tpu=false --precision=float32  2>&1 | tee tpu.log
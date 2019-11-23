#!/bin/bash

for i in {1..102}
do
  j=$[i*1251]
  echo $i
  echo $j
  echo "python ckpt_mask_sparsity.py --checkpoint=/scratch/hj14/tpu_checkpoints/ckpt-102epoch/model.ckpt-$j/model.ckpt-$j \
    --sparsity_technique=magnitude_pruning --model=rn50 | tee -a sparsity-mask-ckpt-102.log " >> mask_sparsity_script.sh
  echo " python resnet_main.py --mode=eval --num_eval_images=64 --eval_batch_size=64 --data_dir=/scratch/hj14/tf_record_latest \
            --model_dir=/scratch/hj14/tpu_checkpoints/ckpt-102epoch/model.ckpt-$j/model.ckpt-$j \
            --use_tpu=false --precision=float32  2>&1 | tee act-sparsity-logs/sparsity-act-ckpt-$j.log " >>  act_sparsity_script.sh
done 

#e.g. 
# python ckpt_mask_sparsity.py --checkpoint=/scratch/hj14/tpu_checkpoints/ckpt-102epoch/model.ckpt-17514/model.ckpt-17514 \
#   --sparsity_technique=magnitude_pruning --model=rn50 | tee -a sparsity-mask-ckpt-102.log 

# python resnet_main.py --mode=eval --num_eval_images=64 --eval_batch_size=64 --data_dir=/scratch/hj14/tf_record_latest \
#             --model_dir=/scratch/hj14/tpu_checkpoints/ckpt-102epoch/model.ckpt-17514/model.ckpt-17514 \
#             --use_tpu=false --precision=float32  2>&1 | tee act-sparsity-logs/sparsity-act-ckpt-17514.log

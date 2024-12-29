TASK_NAME=beer0,beer1,beer2
DATA_NAME=beer
# TASK_NAME=hotel0,hotel1,hotel2
# DATA_NAME=hotel


model_type=mare
use_single_backbone=0
sparsity_control=0.11
remain_layers=9

mare_att=4
num_clusters=3
process_type=2
overall_only=1
mean_first=1
mlp_head=0
annotation_ratio=0
balance=1

SEED=42
MARE_FACTOR=1

export CUDA_VISIBLE_DEVICES=1

if [ $TASK_NAME == 20news ]
then
    EPOCH_NUM=5
else
    EPOCH_NUM=5
fi


OUTPUT_DIR="model/beer_high_sparse"
# OUTPUT_DIR="model/beer_low_sparse"
# OUTPUT_DIR="model/hotel"


python3 mare-master/src/run_classification.py \
  --task_name ${TASK_NAME}  \
  --model_type ${model_type} \
  --mare_coefficient ${MARE_FACTOR} \
  --model_name_or_path $OUTPUT_DIR \
  --data_dir data/${DATA_NAME} \
  --max_seq_length 256  \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --save_steps 500 \
  --learning_rate 3e-5 \
  --num_train_epochs ${EPOCH_NUM}  \
  --output_dir ${OUTPUT_DIR} \
  --do_lower_case  \
  --do_eval \
  --remain_layers $remain_layers \
  --sparsity_control $sparsity_control \
  --use_single_backbone $use_single_backbone \
  --mare_att $mare_att \
  --num_clusters $num_clusters \
  --process_type $process_type \
  --overall_only $overall_only \
  --mean_first $mean_first \
  --mlp_head $mlp_head \
  --annotation_ratio $annotation_ratio \
  --balance $balance \
  --overwrite_output_dir    2>&1 | tee ${OUTPUT_DIR}/log.log
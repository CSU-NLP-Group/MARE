# choose from ['beer0', 'beer1', 'beer2', 'hotel0', 'hotel1', 'hotel2'，'beer_cor_0', 'beer_cor_1', 'beer_cor_2']
TASK_NAME=beer0,beer1,beer2
DATA_NAME=beer
# TASK_NAME=hotel0,hotel1,hotel2
# DATA_NAME=hotel

EPOCH_NUM=5

model_type=mare
model_name_or_path="bert-base-uncased" # choose from ['bert-base-uncased', 'bert-large-uncased']
use_single_backbone=0 # 0: use two separate backbones for rnp; 1: use single backbone for rnp [FR]
share_head=1 # 0: use multiple mare predictors; 1: use single mare predictor
constrain_final=0 # 0: length config; 1: only penalty the final layers

# 0.11,0.12,0.105
# 0.13,0.12,0.11
# 0.19,0.16,0.13
sparsity_control=0.19,0.16,0.13 # sparsity level, 0.1 means 10% of the tokens are remained
remain_layers=9 # number of layers for IG part.
decay_num=0 # number of layers for RG part.
gamma=1 # the initial sparsity level for RG part.
decay_type=log # decay type for RG part, choose from ['log', 'linear', 'exp']
pretrain_steps=0 # number of pretrain steps for skew predictor, 0 means no pretrain.
dataloader_wrapper=balanced_round
multiply_mask=0
mean_bias=5
mare_att=4
num_clusters=3 # num of special tokens
process_type=2 # exdot
overall_only=1
mean_first=1
mlp_head=0 # whether explain head is MLP
balance=1 # 0: do not balance the dataset; 1: balance
cluster_is_cls=2 # 0: MARE-random; 1: MARE-share; 2: MARE-CLS

SEED=00
SPARSE_FACTOR=0.7,3,3 # the coefficient for sparsity control.
CONTIGUOUS_FACTOR=0.7,3,3 # the coefficient for contiguous control.

learning_rate=3e-5

export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR=model/classification/${TASK_NAME}/${model_type}/${model_name_or_path}/$(date +"%m-%d-%H-%M")\
_${SPARSE_FACTOR}_${CONTIGUOUS_FACTOR}_${remain_layers}_${sparsity_control}_${decay_num}_${gamma}\
_${use_single_backbone}_${pretrain_steps}_${share_head}_${constrain_final}\
_${decay_type}_${mean_bias}_${EPOCH_NUM}_${mare_att}_${process_type}_${num_clusters}\
_${overall_only}_${mean_first}_${mlp_head}_${balance}\
_${cluster_is_cls}

mkdir -p ${OUTPUT_DIR}

python mare-master/src/run_classification.py \
  --task_name ${TASK_NAME}  \
  --model_type ${model_type} \
  --sparse_coefficient ${SPARSE_FACTOR} \
  --contiguous_coefficient ${CONTIGUOUS_FACTOR} \
  --model_name_or_path ${model_name_or_path} \
  --data_dir data/${DATA_NAME} \
  --max_seq_length 256  \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --save_steps 1000 \
  --seed $SEED \
  --learning_rate $learning_rate \
  --num_train_epochs ${EPOCH_NUM}  \
  --output_dir ${OUTPUT_DIR} \
  --pretrain_steps ${pretrain_steps} \
  --do_lower_case  \
  --do_eval  \
  --do_train \
  --evaluate_steps 1000 \
  --max_steps -1 \
  --evaluate_during_training \
  --remain_layers $remain_layers \
  --sparsity_control $sparsity_control \
  --use_single_backbone $use_single_backbone \
  --decay_num $decay_num \
  --gamma $gamma \
  --decay_type $decay_type \
  --share_head $share_head \
  --constrain_final $constrain_final \
  --dataloader_wrapper $dataloader_wrapper \
  --multiply_mask $multiply_mask \
  --mean_bias $mean_bias \
  --mare_att $mare_att \
  --process_type $process_type \
  --num_clusters $num_clusters \
  --overall_only $overall_only \
  --mean_first ${mean_first} \
  --mlp_head $mlp_head \
  --balance $balance \
  --cluster_is_cls $cluster_is_cls \
  --overwrite_output_dir    2>&1 | tee ${OUTPUT_DIR}/log.log
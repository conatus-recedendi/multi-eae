#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

if true; then
OUTPUT=wikievents-base
GPU=3
TRAIN_BSZ=4
EVAL_BSZ=2
ACCU=2
LR=3e-5
NOT_BERT_LR=1e-4
LAMBDA_BOUNDARY=0.1
BASE_MODEL=bert-base-uncased
HIDDEN_SIZE=768
POS_LOSS_WEIGHT=10
SPAN_LEN=8
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.05
EVENT_EMBEDDING_SIZE=200
TRAIN_EPOCH=100
seeds=(1000)
EPOCH=50
MAX_LEN=1024
TRAIN_FILE=../../../data/wikievents/transfer-train.jsonl
DEV_FILE=../../../data/wikievents/transfer-dev.jsonl
TEST_FILE=../../../data/wikievents/transfer-test.jsonl
META_FILE=../../../data/wikievents/meta.json
CACHE_DIR=../../../cache/wikievents-base
OUTPUT_DIR=../../../output/wikievents-base_seed${SEED}

# main
for SEED in ${seeds[@]}
do
python train_EAE.py \
--task_name wikievent \
--do_train True \
--do_eval True \
--no_cuda False \
--train_file ${TRAIN_FILE} \
--validation_file ${DEV_FILE} \
--test_file ${TEST_FILE} \
--meta_file ${META_FILE} \
--model_name_or_path ${BASE_MODEL} \
--output_dir ${OUTPUT_DIR} \
--per_device_train_batch_size ${TRAIN_BSZ} \
--per_device_eval_batch_size ${EVAL_BSZ} \
--learning_rate ${LR} \
--not_bert_learning_rate ${NOT_BERT_LR} \
--num_train_epochs ${TRAIN_EPOCH} \
--weight_decay ${WEIGHT_DECAY} \
--remove_unused_columns False \
--load_best_model_at_end \
--metric_for_best_model f1 \
--greater_is_better True \
--evaluation_strategy epoch \
--eval_accumulation_steps 100 \
--logging_strategy epoch \
--warmup_ratio ${WARMUP_RATIO} \
--gradient_accumulation_steps 2 \
--pos_loss_weight ${POS_LOSS_WEIGHT} \
--span_len ${SPAN_LEN} \
--max_len ${MAX_LEN} \
--seed ${SEED} \
--lambda_boundary ${LAMBDA_BOUNDARY} \
--event_embedding_size ${EVENT_EMBEDDING_SIZE} \
--span_len_embedding_range ${HIDDEN_SIZE} \
--span_len_embedding_hidden_size ${HIDDEN_SIZE} \
--cache_dir ${CACHE_DIR}
done
fi

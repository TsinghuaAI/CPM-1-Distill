#! /bin/bash

WORKING_DIR=${HOME}/CPM-distill

# Change for multinode config
MP_SIZE=2

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

DATA_PATH="${WORKING_DIR}/data/corpus_12_1_text_document"

# S_CONFIG_PATH="${WORKING_DIR}/Megatron-LM/configs/model/gpt_medium_config.json"
# S_CKPT_PATH="${WORKING_DIR}/checkpoints/medium/mp2/"

S_CONFIG_PATH="${WORKING_DIR}/Megatron-LM/configs/model/gpt_small_config.json"
S_CKPT_PATH="${WORKING_DIR}/checkpoints/small/mp2/"

# S_CONFIG_PATH="${WORKING_DIR}/Megatron-LM/configs/model/gpt_large_config.json"
# S_CKPT_PATH="/mnt/nfs/home/zzy/checkpoints/CPM-large/"

T_CONFIG_PATH="${WORKING_DIR}/Megatron-LM/configs/model/gpt_large_config.json"
T_CKPT_PATH="/mnt/nfs/home/zzy/checkpoints/CPM-large/"

SAVE_PATH="${WORKING_DIR}/results/"
DS_CONFIG="${WORKING_DIR}/Megatron-LM/configs/deepspeed/ds_zero2_config_small.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_3w_new"
HOST_FILE="${WORKING_DIR}/Megatron-LM/configs/host_files/hostfile-4"

BATCH_SIZE=8
LR=0.00015
TRAIN_ITER=300000
ALPHA_LM=0
ALPHA_CE=1
TEMPERATURE_KD=1

SEQ_LENGTH=1024



GPT_OPT="" 
GPT_OPT+=" --model-parallel-size ${MP_SIZE}"
GPT_OPT+=" --batch-size ${BATCH_SIZE}"
GPT_OPT+=" --seq-length ${SEQ_LENGTH}"
GPT_OPT+=" --train-iters ${TRAIN_ITER}"
GPT_OPT+=" --save ${SAVE_PATH}"
GPT_OPT+=" --student_config_path ${S_CONFIG_PATH}"
GPT_OPT+=" --teacher_config_path ${T_CONFIG_PATH}"
# GPT_OPT+=" --student_load ${S_CKPT_PATH}"
GPT_OPT+=" --teacher_load ${T_CKPT_PATH}"
GPT_OPT+=" --data-path ${DATA_PATH}"
GPT_OPT+=" --data-impl mmap"
GPT_OPT+=" --lazy-loader"
GPT_OPT+=" --tokenizer-type GPT2BPETokenizer"
GPT_OPT+=" --split 949,50,1"
GPT_OPT+=" --distributed-backend nccl"
GPT_OPT+=" --lr ${LR}"
GPT_OPT+=" --no-load-optim"
GPT_OPT+=" --lr-decay-style cosine"
GPT_OPT+=" --weight-decay 1e-2"
GPT_OPT+=" --clip-grad 1.0"
GPT_OPT+=" --warmup 0.01"
GPT_OPT+=" --tokenizer-path ${TOKENIZER_PATH}"
GPT_OPT+=" --save-interval 32000"
GPT_OPT+=" --eval-interval 16000"
GPT_OPT+=" --eval-iters 80"
GPT_OPT+=" --log-interval 80"
GPT_OPT+=" --deepspeed"
GPT_OPT+=" --deepspeed_config ${DS_CONFIG}"
# GPT_OPT+=" --alpha_lm ${ALPHA_LM}"
GPT_OPT+=" --alpha_ce ${ALPHA_CE}"
GPT_OPT+=" --temperature_kd ${TEMPERATURE_KD}"
GPT_OPT+=" --do_train"
GPT_OPT+=" --do_valid"
GPT_OPT+=" --do_test"

# GPT_OPT+=" --checkpoint-activations"
# GPT_OPT+=" --deepspeed-activation-checkpointing"
# GPT_OPT+=" --fp16"

CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} distill_gpt2.py $@ ${GPT_OPT}"
echo ${CMD}
eval ${CMD}

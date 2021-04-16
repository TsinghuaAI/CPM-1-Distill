#! /bin/bash

WORKING_DIR=/mnt/sfs_turbo/cpm-distill

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTIONS_NCCL="NCCL_DEBUG=info"

# Change for multinode config
MP_SIZE=2

DATA_PATH="${WORKING_DIR}/pretrain_data/baike_small_text_document"

S_CONFIG_PATH="${WORKING_DIR}/configs/model/gpt_small_config.json"
S_CKPT_PATH="${WORKING_DIR}/checkpoints/small/mp2/CPM-20000"

T_CONFIG_PATH="${WORKING_DIR}/configs/model/gpt_large_config.json"
T_CKPT_PATH="${WORKING_DIR}/checkpoints/CPM-large/"

SAVE_PATH="${WORKING_DIR}/results/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_zero2_config_small.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_3w_new"

BATCH_SIZE=32
LR=0.00015
TRAIN_ITER=600000
ALPHA_LM=0.6
ALPHA_CE=0.4
TEMPERATURE_KD=1

SEQ_LENGTH=1024

GPT_OPT="" 
GPT_OPT+=" --model-parallel-size ${MP_SIZE}"
GPT_OPT+=" --batch-size ${BATCH_SIZE}"
GPT_OPT+=" --seq-length ${SEQ_LENGTH}"
GPT_OPT+=" --train-iters ${TRAIN_ITER}"
GPT_OPT+=" --save ${SAVE_PATH}"
GPT_OPT+=" --log_file ${LOG_FILE}"
GPT_OPT+=" --student_config_path ${S_CONFIG_PATH}"
GPT_OPT+=" --teacher_config_path ${T_CONFIG_PATH}"
# GPT_OPT+=" --student_load ${S_CKPT_PATH}"
# GPT_OPT+=" --teacher_load ${T_CKPT_PATH}"
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
GPT_OPT+=" --save-interval 10000"
GPT_OPT+=" --eval-interval 2500"
GPT_OPT+=" --eval-iters 10"
GPT_OPT+=" --log-interval 500"
GPT_OPT+=" --deepspeed"
GPT_OPT+=" --deepspeed_config ${DS_CONFIG}"
GPT_OPT+=" --alpha_lm ${ALPHA_LM}"
GPT_OPT+=" --alpha_ce ${ALPHA_CE}"
GPT_OPT+=" --temperature_kd ${TEMPERATURE_KD}"
GPT_OPT+=" --do_train"
GPT_OPT+=" --do_test"
GPT_OPT+=" --teacher_fp16"

GPT_OPT+=" --checkpoint-activations"
GPT_OPT+=" --deepspeed-activation-checkpointing"
GPT_OPT+=" --fp16"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/distill_gpt2.py $@ ${GPT_OPT}"

echo ${CMD}
${CMD}
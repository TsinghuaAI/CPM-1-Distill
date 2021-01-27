#! /bin/bash

# Change for multinode config
MP_SIZE=2

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

# script_path=$(realpath $0)
# script_dir=$(dirname $script_path)

DATA_PATH="/mnt/nfs/home/gyx/CPM-distill/data/corpus_12_1_text_document"
CHECKPOINT_PATH="/mnt/nfs/home/gyx/CPM-distill/results-local/origin/" # 6.4
LOAD_PATH="/mnt/nfs/home/gyx/CPM-distill/checkpoints/small/mp2/my_60000"
# CHECKPOINT_PATH="/mnt/nfs/home/zzy/checkpoints/CPM-medium"

config_json="/mnt/nfs/home/gyx/CPM-distill/Megatron-LM/configs/deepspeed/ds_zero2_config_small.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 4 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --vocab-size 30000 \
       --train-iters 300000 \
       --save $CHECKPOINT_PATH \
       --load ${LOAD_PATH} \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --tokenizer-path /mnt/nfs/home/gyx/CPM-distill/bpe_3w_new \
       --save-interval 32000 \
       --eval-interval 16000 \
       --eval-iters 80 \
       --log-interval 80 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"



run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile /mnt/nfs/home/gyx/CPM-distill/Megatron-LM/configs/host_files/hostfile-6 pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

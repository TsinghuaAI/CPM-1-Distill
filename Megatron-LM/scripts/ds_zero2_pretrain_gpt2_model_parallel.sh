#! /bin/bash

# Change for multinode config
MP_SIZE=2

NUM_WORKERS=8
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

DATA_PATH="/mnt/nfs/home/zzy/data/bpe_3w_100G_text_document"
CHECKPOINT_PATH="/mnt/nfs/home/zzy/checkpoints/3B-new-bpe-fat"

config_json="$script_dir/ds_zero2_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 32 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --batch-size 24 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --vocab-size 30000 \
       --train-iters 300000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
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
       --tokenizer-path bpe_chinese/ \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
       --save-interval 2000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --log-interval 10 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile /mnt/nfs/home/zzy/hostfile pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

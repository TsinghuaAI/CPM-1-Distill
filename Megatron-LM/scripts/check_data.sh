#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

DATA_PATH="/mnt/nfs/home/zgy/deepspeed-gpt/corpus_10_30_text_document"
#DATA_PATH="/mnt/nfs/home/zzy/data/bpe_3w_100G_text_document"
CHECKPOINT_PATH="/mnt/nfs/home/zzy/checkpoints/test"

config_json="$script_dir/ds_zero2_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 6 \
       --hidden-size 128 \
       --num-attention-heads 4 \
       --batch-size 12 \
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


run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} check_data.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

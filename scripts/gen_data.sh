WORKING_DIR=${YOUR_PATH_TO}/CPM-Distill

python3 ${WORKING_DIR}/tools/preprocess_data.py --input ${YOUR_PATH_TO}/data/train.txt --tokenizer-path ${WORKING_DIR}/bpe_3w_new/ --output-prefix ${WORKING_DIR}/pretrain_data/distill_pretrain_data --workers 30
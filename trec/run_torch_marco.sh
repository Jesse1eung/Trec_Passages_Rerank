#!/bin/bash
# MODEL_DIR=../data/pytorch_bert_base
MODEL_DIR=../model/bert-base-uncased

python3 ./run.py \
    --train_file=./data/train.tsv \
    --eval_file=./data/eval.tsv \
    --model_name_or_path=${MODEL_DIR}/pytorch_model.bin \
    --config_name=${MODEL_DIR}/bert_config.json \
    --vocab_file=${MODEL_DIR}/vocab.txt \
    --max_query_len=64 \
    --max_seq_len=512 \
    --do_eval \
    --num_training_steps=100000 \
    --eval_batch_size=64 \
    --learning_rate=3e-6

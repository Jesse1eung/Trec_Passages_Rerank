import argparse
import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertTokenizer, BertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from model import BertForPassageRerank

from utils import PassageData
from utils import my_collate_fn
from torch.utils.data import DataLoader

import tokenization

# python3 ./run.py \
#     --train_file='./data/train.tsv' \
#     --model_name_or_path='../model/bert-base-uncased/pytorch_model.bin' \
#     --config_name='../model/bert-base-uncased/bert_config.json' \
#     --vocab_file='../model/bert-base-uncased/vocab.txt' \
#     --max_query_len=64 \
#     --max_seq_len=512 \
#     --do_train \
#     --num_training_steps=100000 \
#     --train_batch_size=16 \
#     --learning_rate=3e-6


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--eval_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model name")
    parser.add_argument("--vocab_file", default="", type=str,
                        help="vocab file path if not the same as model name")
    # parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--max_query_len", default=64, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--num_training_steps", default=10000, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    params = {'batch_size': args.train_batch_size,
              'shuffle': True,
              'num_workers': 8,
              'collate_fn': my_collate_fn}

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=True)
    config = BertConfig.from_pretrained(
        args.config_name, num_labels=args.num_labels)
    config.num_labels = 2
    print((config))
    model = BertForPassageRerank.from_pretrained(
        args.model_name_or_path, config=config)

    model.to(args.device)

    if args.do_train:
        print("training...")
        params = {'batch_size': args.train_batch_size,
                  'shuffle': True,
                  'num_workers': 2,
                  'collate_fn': my_collate_fn}
        # tokenizer = BertTokenizer.from_pretrained(
        #   args.tokenizer_name, do_lower_case=True)

        train_set = PassageData(args.train_file, tokenizer,
                                args.max_query_len, args.max_seq_len)
        dataloader = DataLoader(train_set, **params)
        num_train_each_epoch = len(dataloader)
        print("step: ", num_train_each_epoch)
        num_training_steps = len(dataloader) * args.epoch
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps)
        for ep in tqdm(range(args.epoch)):

            running_loss = 0.0
            step = 0
            for data in tqdm(dataloader):
                step += 1
                # data = data.to(args.device)
                inputs_ids, masks, segments_ids, \
                    labels = [x.to(args.device) for x in data]
                outputs = model(inputs_ids, masks, segments_ids, labels)
                loss = outputs

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 500 == 0:
                    print("loss:", loss)
                running_loss += loss.item()

    elif args.do_eval:

        print("evaling...")
        model.eval()

        idmap = "./data/ids_map.json"
        with open(idmap, 'r') as f:
            id_map = json.load(f)
        q_ids = id_map['q_id']
        qid_to_pid = id_map['qid_to_pid']
        params = {'batch_size': args.eval_batch_size,
                  'shuffle': False,
                  'num_workers': 4,
                  'collate_fn': my_collate_fn}
        eval_set = PassageData(args.eval_file, tokenizer,
                               args.max_query_len, args.max_seq_len)
        dataloader = DataLoader(eval_set, **params)
        results = []
        count = 0
        fw = open('./data/output.tsv', 'w')
        i = 0
        for data in dataloader:
            if count == 2:
                break
            i += 1
            print(i)
            inputs_ids, masks, \
                segments_ids = [x.to(args.device) for x in data]
            with torch.no_grad():
                result = model(inputs_ids, masks, segments_ids)
            # print("result: ", result)

            for res in result:
                results.append(res[1])
                if len(results) == 1000:
                    print('greater than 1000')
                    q_id = q_ids[count]
                    pred_passages = torch.argsort(
                        result[:, 1], descending=True, dim=-1)
                    rank = 1
                    for idx in pred_passages:
                        p_id = qid_to_pid[q_id][idx.item()]
                        if p_id != '000000':
                            fw.write(q_id + '\t' + p_id +
                                     '\t' + str(rank + 1) + '\n')
                            rank += 1
                    count += 1
                    results = []
        fw.close()


if __name__ == '__main__':
    main()

import torch
import pandas as pd
import numpy as np
# 数据格式 [query, pos_ans, neg_ans]

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PassageData(Dataset):
    def __init__(self, path_to_file, tokenizer, max_query_len, max_seq_len):
        self.df = pd.read_csv(path_to_file, sep='\t', names=[
                              'query', 'passage', 'label'])
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len

        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids, segments_ids, label = self.get_ids(row)
        return (input_ids, segments_ids, label)

    def __len__(self):
        return len(self.df)

    def get_ids(self, row):
        query_tokens = self.tokenizer.tokenize(row['query'])
        if len(query_tokens) > self.max_query_len - 2:
            query_tokens = query_tokens[:self.max_query_len - 2]

        word_pieces = ["[CLS]"]
        word_pieces += query_tokens + ["[SEP]"]
        len_query = len(word_pieces)

        passage_tokens = self.tokenizer.tokenize(row['passage'])
        if len(passage_tokens) > self.max_seq_len - len_query - 1:
            passage_tokens = passage_tokens[:self.max_seq_len - len_query - 1]

        word_pieces += passage_tokens + ["[SEP]"]
        len_passage = len(word_pieces) - len_query

        input_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segments_ids = torch.tensor(
            [0] * len_query + [1] * len_passage, dtype=torch.long)
        label = row['label']

        return (input_ids, segments_ids, label)


def my_collate_fn(batch):
    inputs_ids = [x[0] for x in batch]
    segments_ids = [x[1] for x in batch]
    # print("shape of batch", len(batch[0]))
    labels = None
    if not np.isnan(batch[0][2]):
        labels = [x[2] for x in batch]
        labels = torch.tensor(labels, dtype=torch.long)

    # pad_sequence的输入是[tensor1, tensor2, ...]
    inputs_ids = pad_sequence(inputs_ids, batch_first=True)
    # inputs_ids = torch.stack(inputs_ids)

    segments_ids = pad_sequence(segments_ids, batch_first=True)
    # segments_ids = torch.stack(segments_ids)

    masks = torch.zeros_like(inputs_ids, dtype=torch.long)
    masks = masks.masked_fill(inputs_ids != 0, 1)

    return (inputs_ids, masks, segments_ids, labels) if labels else (inputs_ids, masks, segments_ids,)

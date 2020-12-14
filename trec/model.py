import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertForPassageRerank(BertModel):
    def __init__(self, config):
        super(BertForPassageRerank, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print("output's shape: ", output.shape)
        # encoder_outputs = tuple(
        # 						v
        # 						for v in (hidden_states, all_hidden_states, all_self_attentions,...)
        # 						if v is not None
        # )
        # output = (sequence_output, pooled_output) + encoder_outputs[1:]
        # pooler_output shape --> (bs, d_embedding)
        pooled_output = output.pooler_output

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        log_probs = F.log_softmax(logits, dim=-1)
        # loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1))
            return loss
        return log_probs
        # return ((loss, ) + logits) if loss is not None else logits

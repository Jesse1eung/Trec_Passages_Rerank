import os
import json
import collections
import pandas as pd
# file_name = "./data/triples.train.small.tsv"

# with open(file_name, 'r') as f:
#     line = f.readline()
#     # while line:

# df = pd.read_csv(file_name, sep='\t', names=[
#                  'query', 'pos_passage', 'neg_passage'])

# result = pd.DataFrame(columns=('query', 'passage', 'label'))

# # print
# type_passage = {'neg_passage': 0, 'pos_passage': 1}
# for i in range(df.shape[0]):

#     for pas in type_passage:
#         result.loc[2 * i + type_passage[pas], 'query'] = df.loc[i, 'query']
#         result.loc[2 * i + type_passage[pas], 'passage'] = df.loc[i, pas]
#         result.loc[2 * i + type_passage[pas], 'label'] = type_passage[pas]


# print(result.describe())
# result.to_csv("./data/train.tsv", sep='\t', index=None, header=0)
file_name = "./data/top1000.dev.tsv"
file_eval = "./data/eval.tsv"

q_id = []
qid_to_pid = collections.defaultdict(list)
with open(file_eval, 'w') as fw:
    with open(file_name, 'r') as f:
        pre = []
        j = 0
        line = f.readline()
        while j < 43:
            # line = f.readline()
            pre = line.split('\t')
            q_id.append(pre[0])
            print(len(q_id))
            for i in range(1000):
                if i > 0 and line.split('\t')[0] != pre[0]:
                    content = pre[2] + '\tFake Document\n'
                    qid_to_pid[q_id[-1]].append('000000')
                    fw.write(content)

                else:

                    qid_to_pid[q_id[-1]].append(line.split('\t')[1])
                    fw.write("\t".join(line.split('\t')[2:]))
                    line = f.readline()

            j += 1
w = json.dumps({'q_id': q_id, 'qid_to_pid': qid_to_pid}, indent=4)
with open("./data/ids_map.json", 'w') as fw:
    fw.write(w)

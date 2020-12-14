
import os
import pandas as pd
file_name = "./data/triples.train.small.tsv"

with open(file_name, 'r') as f:
    line = f.readline()
    # while line:

df = pd.read_csv(file_name, sep='\t', names=[
                 'query', 'pos_passage', 'neg_passage'])

result = pd.DataFrame(columns=('query', 'passage', 'label'))

# print
type_passage = {'neg_passage': 0, 'pos_passage': 1}
for i in range(df.shape[0]):

    for pas in type_passage:
        result.loc[2 * i + type_passage[pas], 'query'] = df.loc[i, 'query']
        result.loc[2 * i + type_passage[pas], 'passage'] = df.loc[i, pas]
        result.loc[2 * i + type_passage[pas], 'label'] = type_passage[pas]


print(result.describe())
result.to_csv("./data/train.tsv", sep='\t', index=None, header=0)

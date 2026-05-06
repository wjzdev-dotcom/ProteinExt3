import pandas as pd
import gzip
from datetime import datetime

CUTOFF = '2022-01-01'

# 读取注释
df = pd.read_csv('./sprot_annotations.tsv', sep='\t')
df['date'] = pd.to_datetime(df['date'])

# 每个蛋白取最早日期
first_date = df.groupby('EntryID')['date'].min()
train_ids = set(first_date[first_date < CUTOFF].index)
test_ids = set(first_date[first_date >= CUTOFF].index)

print(f"训练集: {len(train_ids)} 个蛋白")
print(f"测试集: {len(test_ids)} 个蛋白")

# 读取 fasta 序列
sequences = {}
current_id = None
current_seq = []
with gzip.open('./training/data/raw/uniprot_sprot.fasta.gz', 'rt') as f:
    for line in f:
        if line.startswith('>'):
            if current_id:
                sequences[current_id] = ''.join(current_seq)
            current_id = line[1:].strip().split('|')[1] if '|' in line else line[1:].strip().split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id:
        sequences[current_id] = ''.join(current_seq)

print(f"fasta 序列数: {len(sequences)}")

# 写出训练集 fasta
import os
os.makedirs('./training/data/raw', exist_ok=True)

with open('./training/data/raw/training.fasta', 'w') as f:
    for pid in train_ids:
        if pid in sequences:
            f.write(f">{pid}\n{sequences[pid]}\n")

# 写出测试集 fasta（用于 MMseqs2 去重）
with open('./test_raw.fasta', 'w') as f:
    for pid in test_ids:
        if pid in sequences:
            f.write(f">{pid}\n{sequences[pid]}\n")

# 写出训练集注释
train_anno = df[df['EntryID'].isin(train_ids)][['EntryID', 'term', 'aspect']]
train_anno.to_csv('./training/data/raw/training.tsv', sep='\t', index=False)

print("Done")

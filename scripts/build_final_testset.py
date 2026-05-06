import pandas as pd
import re

pattern = re.compile(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$')
CUTOFF = '20230101'

# 读取同源蛋白
hits = pd.read_csv('./test_vs_train2.tsv',
                   sep='\t', header=None, names=['query','target','pident'])
homologous = set(hits[hits['pident'] >= 50]['query'])

# 读取注释
pid_date = {}
pid_annotations = {}
with open('/home/jingzhi/goa_exp_annotations.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        pid, go, evidence, aspect, date = parts[0], parts[1], parts[2], parts[3], parts[4]
        if not pattern.match(pid) or '_' in pid or '-' in pid:
            continue
        if pid not in pid_date or date < pid_date[pid]:
            pid_date[pid] = date
        if pid not in pid_annotations:
            pid_annotations[pid] = []
        pid_annotations[pid].append((go, aspect, date))

# 测试集：2023年后，非同源
test_ids = {pid for pid, d in pid_date.items() 
            if d >= CUTOFF and pid not in homologous}
print(f"最终测试集蛋白数: {len(test_ids)}")

# 读取序列
sequences = {}
current_id = None
current_seq = []
with open('./all_sequences.fasta') as f:
    for line in f:
        if line.startswith('>'):
            if current_id:
                sequences[current_id] = ''.join(current_seq)
            pid = line[1:].strip().split('|')[1] if '|' in line else line[1:].strip().split()[0]
            current_id = pid
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id:
        sequences[current_id] = ''.join(current_seq)

# 写出测试集 fasta
import os
os.makedirs('./comparison/fasta/test', exist_ok=True)
with open('./comparison/fasta/test/test.fasta', 'w') as f:
    count = 0
    for pid in test_ids:
        if pid in sequences:
            f.write(f">{pid}\n{sequences[pid]}\n")
            count += 1
print(f"写出序列: {count} 个")

# 写出测试集注释（CAFA 格式）
rows = []
for pid in test_ids:
    for go, aspect, date in pid_annotations.get(pid, []):
        aspect_map = {'F': 'F', 'P': 'P', 'C': 'C'}
        if aspect in aspect_map:
            rows.append({'EntryID': pid, 'term': go, 'aspect': aspect_map[aspect]})

anno_df = pd.DataFrame(rows).drop_duplicates()
anno_df.to_csv('./comparison/fasta/test/test.tsv', sep='\t', index=False)
print(f"写出注释行数: {len(anno_df)}")
print("Done")

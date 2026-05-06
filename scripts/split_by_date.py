import re

pattern = re.compile(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$')
CUTOFF = '20230101'

# 读取每个蛋白的最早注释日期
pid_date = {}
with open('/home/jingzhi/goa_exp_annotations.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        pid, date = parts[0], parts[4]
        if not pattern.match(pid) or '_' in pid or '-' in pid:
            continue
        if pid not in pid_date or date < pid_date[pid]:
            pid_date[pid] = date

train_ids = {pid for pid, d in pid_date.items() if d < CUTOFF}
test_ids = {pid for pid, d in pid_date.items() if d >= CUTOFF}
print(f"训练集: {len(train_ids)} 个蛋白")
print(f"测试集: {len(test_ids)} 个蛋白")

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

print(f"序列总数: {len(sequences)}")

# 写出训练集
import os
os.makedirs('./training/data/raw', exist_ok=True)
with open('./training/data/raw/training.fasta', 'w') as f:
    for pid in train_ids:
        if pid in sequences:
            f.write(f">{pid}\n{sequences[pid]}\n")

# 写出测试集 raw（用于 MMseqs2）
with open('./test_raw.fasta', 'w') as f:
    for pid in test_ids:
        if pid in sequences:
            f.write(f">{pid}\n{sequences[pid]}\n")

print(f"训练集写出: {sum(1 for p in train_ids if p in sequences)} 个")
print(f"测试集写出: {sum(1 for p in test_ids if p in sequences)} 个")
print("Done")

import gzip
import csv
from datetime import datetime

# 实验证据代码
EXP_CODES = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 
             'HTP', 'HDA', 'HMP', 'HGI', 'HEP',
             'TAS', 'IC'}

dat_path = './training/data/raw/uniprot_sprot.dat.gz'

proteins = []
current = {}
go_terms = []

print("解析 Swiss-Prot dat 文件...")
with gzip.open(dat_path, 'rt', encoding='utf-8') as f:
    for line in f:
        tag = line[:2].strip()
        content = line[5:].strip()

        if tag == 'AC' and 'id' not in current:
            current['id'] = content.split(';')[0].strip()

        elif tag == 'DT' and 'date' not in current:
            # 格式: 01-JAN-2023, integrated into UniProtKB/Swiss-Prot.
            if 'integrated' in content:
                try:
                    date_str = content.split(',')[0].strip()
                    current['date'] = datetime.strptime(date_str, '%d-%b-%Y')
                except:
                    pass

        elif tag == 'DR' and content.startswith('GO;'):
            parts = content.split(';')
            if len(parts) >= 4:
                go_id = parts[1].strip()
                evidence = parts[3].strip().split(':')[0].strip()
                aspect_char = parts[2].strip()[0]  # C, F, P
                aspect_map = {'C': 'C', 'F': 'F', 'P': 'P'}
                aspect = aspect_map.get(aspect_char)
                if aspect and evidence in EXP_CODES:
                    go_terms.append((go_id, aspect, evidence))

        elif tag == '//':
            if 'id' in current and 'date' in current and go_terms:
                for go_id, aspect, evidence in go_terms:
                    proteins.append({
                        'EntryID': current['id'],
                        'date': current['date'],
                        'term': go_id,
                        'aspect': aspect,
                        'evidence': evidence
                    })
            current = {}
            go_terms = []

print(f"解析完成，总记录数: {len(proteins)}")

# 写出
import pandas as pd
df = pd.DataFrame(proteins)
print(f"唯一蛋白数: {df['EntryID'].nunique()}")
print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
df['date'] = df['date'].dt.strftime('%Y-%m-%d')
df.to_csv('./sprot_annotations.tsv', sep='\t', index=False)
print("Done")

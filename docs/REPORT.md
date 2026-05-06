# Comparison Report

## MFO (F)

| Method | Fmax | Smin | AUPR | AvgAUC |
| --- | --- | --- | --- | --- |
| BLAST | 0.3224 | 11.7482 | 0.1470 | 0.6233 |
| DeepGO-SE | 0.5751 | 11.5543 | 0.5118 | 0.6192 |
| ESM2(8M)+MLP | 0.6418 | 12.0407 | 0.6091 | 0.7117 |
| Naive | 0.6560 | 12.3815 | 0.2407 | 0.5000 |
| ProteinExt3.1 | 0.6688 | 10.1209 | 0.6192 | 0.7724 |
| ProteinExt3.1-meanPooling | 0.6251 | 10.3121 | 0.5470 | 0.7577 |
| ProteinExt3.1-noBLAST | 0.6739 | 10.1415 | 0.6171 | 0.7760 |
| ProteinExt3.1-noCrafted | 0.6427 | 10.3345 | 0.5687 | 0.7642 |
| TALE | 0.4425 | 13.1036 | 0.3512 | 0.6034 |

## BPO (P)

| Method | Fmax | Smin | AUPR | AvgAUC |
| --- | --- | --- | --- | --- |
| BLAST | 0.1277 | 63.2775 | 0.0270 | 0.5549 |
| DeepGO-SE | 0.1891 | 61.9634 | 0.0750 | 0.5513 |
| ESM2(8M)+MLP | 0.1259 | 63.5054 | 0.0454 | 0.6955 |
| Naive | 0.0199 | 64.9770 | 0.0064 | 0.5000 |
| ProteinExt3.1 | 0.2258 | 60.7704 | 0.1228 | 0.7395 |
| ProteinExt3.1-meanPooling | 0.2239 | 60.1319 | 0.1189 | 0.7248 |
| ProteinExt3.1-noBLAST | 0.2207 | 60.4412 | 0.1207 | 0.7510 |
| ProteinExt3.1-noCrafted | 0.2159 | 60.2661 | 0.1152 | 0.7304 |
| TALE | 0.1234 | 63.8433 | 0.0378 | 0.5228 |

## CCO (C)

| Method | Fmax | Smin | AUPR | AvgAUC |
| --- | --- | --- | --- | --- |
| BLAST | 0.3526 | 15.4196 | 0.1598 | 0.5803 |
| DeepGO-SE | 0.5218 | 13.6258 | 0.4333 | 0.5959 |
| ESM2(8M)+MLP | 0.4991 | 14.5133 | 0.4079 | 0.7305 |
| Naive | 0.4470 | 14.6050 | 0.2329 | 0.5000 |
| ProteinExt3.1 | 0.5467 | 13.3046 | 0.4897 | 0.7936 |
| ProteinExt3.1-meanPooling | 0.4914 | 14.4215 | 0.4228 | 0.7533 |
| ProteinExt3.1-noBLAST | 0.5227 | 13.7174 | 0.4711 | 0.7895 |
| ProteinExt3.1-noCrafted | 0.5069 | 14.0217 | 0.4469 | 0.7593 |
| TALE | 0.4477 | 15.2306 | 0.3748 | 0.5929 |

## Fmax Thresholds

| Method | MFO | BPO | CCO |
| --- | --- | --- | --- |
| BLAST | 0.19 | 0.07 | 0.14 |
| DeepGO-SE | 0.42 | 0.16 | 0.29 |
| ESM2(8M)+MLP | 0.13 | 0.03 | 0.08 |
| Naive | 0.04 | 0.02 | 0.10 |
| ProteinExt3.1 | 0.15 | 0.07 | 0.12 |
| ProteinExt3.1-meanPooling | 0.19 | 0.10 | 0.12 |
| ProteinExt3.1-noBLAST | 0.18 | 0.09 | 0.13 |
| ProteinExt3.1-noCrafted | 0.20 | 0.11 | 0.14 |
| TALE | 0.49 | 0.16 | 0.27 |

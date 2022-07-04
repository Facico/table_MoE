import json
import csv
import pandas as pd

if __name__ == '__main__':
    tsv_path = "/data2/private/fanchenghao/DPR/downloads/psgs_w100.tsv"
    train_df = pd.read_csv(open(tsv_path, 'r'), delimiter='\t')
    n = len(train_df)
    for i in range(20):
        for j in range(n // 20 * i, min(n, n // 20 * (i + 1))):
            
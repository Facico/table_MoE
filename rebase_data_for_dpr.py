import json
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    dpr_qas_train_path = '/data3/private/fanchenghao/DPR/downloads/data/retriever/qas/nq-train.csv'
    dpr_qas_dev_path = '/data3/private/fanchenghao/DPR/downloads/data/retriever/qas/nq-dev.csv'
    dpr_qas_test_path = '/data3/private/fanchenghao/DPR/downloads/data/retriever/qas/nq-test.csv'

    nq_origin_train_path = '/data3/private/fanchenghao/OTT-QA/table_crawling/data/train_qu.json'
    nq_origin_dev_path = '/data3/private/fanchenghao/OTT-QA/table_crawling/data/dev_qu.json'

    train_df = pd.read_csv(open(dpr_qas_train_path, 'r'), delimiter='\t')
    
    dpr_train_dict = {train_df.columns[0] : 1}
    tot_train = 1
    for i in range(len(train_df)):
        #print(train_df.iloc[i, 0])
        if train_df.iloc[i, 0] in dpr_train_dict:
            print('chong fu le', train_df.iloc[i, 0], train_df.iloc[i, 1], '&', dpr_train_dict[train_df.iloc[i, 0]])
        dpr_train_dict[train_df.iloc[i, 0]] = train_df.iloc[i, 1]
        tot_train += 1
    
    print(tot_train)

    nq_train_dict = json.load(open(nq_origin_train_path, 'r'))
    nq_dev_dict = json.load(open(nq_origin_dev_path, 'r'))

    nq_all_dict = {}
    num_all = 0
    chong_fu = 0
    for k, v in nq_train_dict.items():
        if k not in nq_all_dict:
            nq_all_dict[k] = v
            num_all += 1
        else:
            chong_fu += 1
    for k, v in nq_dev_dict.items():
        if k not in nq_all_dict:
            nq_all_dict[k] = v
            num_all += 1
        else:
            chong_fu += 1
    print(num_all, chong_fu)

    found = 0
    not_found = 0
    found_in_dev = 0
    for k, v in dpr_train_dict.items():
        if k not in nq_train_dict:
            not_found += 1
            if k in nq_dev_dict:
                found_in_dev += 1
        else:
            found += 1
    
    print('train', found, not_found, found_in_dev)

    found = 0
    not_found = 0
    for k, v in dpr_train_dict.items():
        if k not in nq_dev_dict:
            not_found += 1
        else:
            found += 1
    print('dev', found, not_found)
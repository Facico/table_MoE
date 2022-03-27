import os
import json
from tqdm import tqdm
import pandas as pd

def add_new(table, table_csv, table_id):
    table_id += 1
    table_csv = table_csv.append({'id': table['id'], 'text': table['text'], 'title': table['title']}, ignore_index=True)
    return table_csv, table_id
if __name__ == '__main__':
    file_path = '/data2/private/fanchenghao/DPR/downloads/data/retriever'
    table_path = "/data2/private/fanchenghao/tapas/tables/tabfact_process_all.json"
    csv_path = os.path.join(file_path, 'table_mine_list.tsv')
    table_set = {}
    table_id = -1
    table_csv = pd.DataFrame(columns=['id', 'text', 'title'])
    table_dict = {}
    with open(table_path, 'r') as f:
        table_dict = json.load(f)
    for k, v in table_dict.items():
        tablex = {'id': k, 'text': v['mine_set'], 'title': v['documentTitle']}
        table_csv, table_id = add_new(tablex, table_csv, table_id)
    """ for file in os.listdir(file_path):
        file_path_x = os.path.join(file_path, file)  
        #print(file)
        if ('table_train_' in file) or ('table_dev' in file):
            print(file_path_x)
            datax = json.load(open(file_path_x, 'r'))
            for i in tqdm(datax):
                for j in i['positive_ctxs']:
                    if j['title'] not in table_set:
                        table_set[j['title']] = 1
                        table_csv, table_id = add_new(j, table_csv, table_id)
                for j in i['negative_ctxs']:
                    if j['title'] not in table_set:
                        table_set[j['title']] = 1
                        table_csv, table_id = add_new(j, table_csv, table_id)"""
        
        
    table_csv.to_csv(csv_path, sep='\t', index=False)
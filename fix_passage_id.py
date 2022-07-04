from ipaddress import v4_int_to_packed
from rank_bm25 import BM25Okapi
import json
import os
import jsonlines
from tqdm import tqdm
import numpy as np
import gzip
import time
import jsonlines
import re
import pandas as pd
import multiprocessing
import csv

def deal_str(str_x):
    str_x = str_x.replace(' ', '')
    str_x = str_x.replace('\n', '')
    return str_x.lower()
if __name__ == '__main__':
    step = ['1', '2']
    json_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/nq_raw_tables/nq-train_raw_table_pos_dpr_neg.json'  
    result_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/all_raw_table_chunks.tsv'
    all_table_file = '/data2/private/fanchenghao/UDT-QA/downloads/data/all_raw_table_chunks_for_index.json'
    """ csv.field_size_limit(500 * 1024 * 1024)
    csv_reader = csv.reader(open(result_path, 'r'), delimiter='\t')
    title_dict = {}
    for row in csv_reader:
        if(row[0] == 'id'): continue
        title = row[2]
        if title not in title_dict:
            title_dict[title] = []
        title_dict[title].append({'id': row[0], 'text':row[1]})

    with open('/data2/private/fanchenghao/UDT-QA/downloads/data/title_dict.json', 'w') as f:
        json.dump(title_dict, f)"""
    
    #with open('/data2/private/fanchenghao/UDT-QA/downloads/data/title_dict.json', 'r') as f:
    #    title_dict = json.load(f)

    with open(all_table_file, 'r') as f:
        data = json.load(f)
    
    id_dict = {}
    for i in tqdm(range(len(data))):
        id_dict[data[i]['chunk_id']] = data[i]['text']
    
    wu = 0
    with open(json_path, 'r') as f:
        data = json.load(f)
        for i in tqdm(range(len(data))):
            """for j in range(len(data[i]['positive_ctxs'])):
                pos_x = data[i]['positive_ctxs'][j]
                #print(pos_x['text'])
                id = 0
                for passage in title_dict[pos_x['title']]:
                    #print(passage['text'])
                    if deal_str(passage['text']) == deal_str(pos_x['text']):
                        id = passage['id']
                        break
                data[i]['positive_ctxs'][j]['passage_id'] = id
                if id == 0:
                    wu += 1"""

            for j in range(len(data[i]['negative_ctxs'])):
                neg_x = data[i]['negative_ctxs'][j]
                id = 0
                if neg_x['id'] in id_dict:
                    pass
                    """print(neg_x['id'])
                    print(id_dict[neg_x['id']])
                    print(neg_x['text'])"""
                    #break
                else:
                    print('id is:', neg_x['id'])
                    break
                """for passage in title_dict[neg_x['title']]:
                    if deal_str(passage['text']) == deal_str(neg_x['text']):
                        id = passage['id']
                        break"""
                """data[i]['negative_ctxs'][j]['passage_id'] = id
                if id == 0:
                    wu += 1
                    if(wu == 1):
                        print(data[i]['negative_ctxs'][j]['text'])
                        for x in title_dict[neg_x['title']]:
                            print(x['text'])"""
            #break
    
    #print(wu)

    """with open('/data2/private/fanchenghao/UDT-QA/downloads/data/nq-train_raw_table_pos_bm25_neg_have_id.json', 'w') as f:
        json.dump(data, f)"""
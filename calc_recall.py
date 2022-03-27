import json
import os
from tqdm import tqdm
import time
import gzip
from multiprocessing import Pool
import multiprocessing
import re
from functools import partial
import sys
import jsonlines

import urllib.parse
import re
import urllib3
# HTTP manager
http = urllib3.PoolManager()
urllib3.disable_warnings()

def url2dockey(string):
    string = urllib.parse.unquote(string)
    string = re.sub('/', '_', string)
    string = re.sub(r'\\', '_', string)
    return string

if __name__ == '__main__':

    bm25_result_path = '/data2/private/fanchenghao/DPR/outputs/retrieve_train_out.json'
    with open(bm25_result_path, 'r') as f:
        bm25_result = json.load(f)
    id_to_table_dict_path = '/data2/private/fanchenghao/tapas/tables/id_to_table.json'
    with open(id_to_table_dict_path, 'r') as f:
        id_to_table_dict = json.load(f)
    test_jsonl_path = '/data2/private/fanchenghao/tapas/interactions/test.jsonl'
    positive_one = {}
    query2id = {}
    with jsonlines.open(test_jsonl_path, 'r') as f:
        print('read test file ...')
        for i in tqdm(f):
            query = i["questions"][0]['originalText']
            if('’' in query):
                print('\u2019' in query)
                query = query.replace('\u2019', '\'')
                print(query)
            query2id[query] = i["questions"][0]['id']
            if "alternativeTableIds" in i["table"]["tableId"]:
                positive_one[i['id']] = [i["table"]["tableId"]] + i["table"]["alternativeTableIds"]
            else:
                positive_one[i['id']] = [i["table"]["tableId"]]
    
    AP_id = [1, 5, 10, 20, 50, 100]
    for j in AP_id:
        r = 0
        tot = 0
        for i in range(len(bm25_result)):
            query = bm25_result[i]['question']
            #print(query)
            """if('\'' in query and b'\\\u2019' not in query.encode('unicode-escape')):
                print(query.encode('unicode-escape'))
                query = query.replace('\'', '\u2019')
                print(query.encode('unicode-escape'))"""
            example_id = query2id[query]
                
            #query_id = example_id
            query_id = '_'.join(example_id.split('_')[:-1])
            positive_list = []
            for o in positive_one[query_id]:
                positive_list.append(id_to_table_dict[o])
            
            yes = 0
            for k in range(j):
                answer_doc_file_name = bm25_result[i]['ctxs'][k]["id"]
                if answer_doc_file_name in positive_list:
                    yes += 1
            
            r += yes / len(positive_list)
            tot += 1
        print('R@{}: {:.3f}'.format(j, r / tot))
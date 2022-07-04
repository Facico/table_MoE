from ipaddress import v4_int_to_packed
from rank_bm25 import BM25Okapi
import json
import os
import jsonlines
from tqdm import tqdm
import numpy as np
import gzip
from multiprocessing import Pool
import multiprocessing
import time
import jsonlines
import re

def search_bm25(tokenized_query, id, bm25, k=100):
    #tokenized_query = x[0]
    #i_query = x[1]
    #bm25 = x[2]
    result = {}
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:k]
    score = [scores[i] for i in top_n]
    result[id] = {'candidate_doc': [corpus_name[i] for i in top_n], 'score': score}
    return result

def file2query(filename):
    query = []

    print('read file...')
    with jsonlines.open(filename, 'r') as f:
        for i in tqdm(f):
            query.append(i)
    print('{} have {} data'.format(filename, str(len(query))))
    return query

if __name__ == '__main__':
    cores = 4#multiprocessing.cpu_count()
    pool = Pool(cores)

    tabfact_file_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process/'   #2366545
    corpus = []

    corpus_name = []
    tot = 0
    table_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process/'
    print('load table ...')
    table_dict = {}
    if os.path.exists('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json') == True:
        table_tab = json.load(open('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json', 'r'))
        for k, v in table_tab.items():
            corpus.append(v['tapas_set'])
            corpus_name.append(v['file_name'])
    else:
        assert('file not found error')
    """for file in tqdm(os.listdir(tabfact_file_path)):
        file_path = os.path.join(tabfact_file_path, file)
        tot += 1

        with open(file_path, 'r') as f:
            table = json.load(f)
            #corpus.append(table['table_text_horizontal'])
            corpus.append(table['tapas_set'])
            corpus_name.append(table['file_name'])"""

    print('all corpus: {}'.format(len(corpus)))

    tokenized_corpus = [re.split(r'\s+', doc) for doc in corpus]``

    bm25 = BM25Okapi(tokenized_corpus)

    
    file_list = ['/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_train.jsonl', '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_dev.jsonl', '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_test.jsonl']
    output_type = ['train', 'dev', 'test']
    for j, file_tdt in enumerate(file_list):
        print(file_tdt)
        query = file2query(file_tdt)
        tokenized_query = [(re.split(r'\s+', doc["question"]) + re.split(r'\s+', doc["question"])[2:] + re.split(r'\s+', doc["question"])[2:] + re.split(r'\s+', doc["question"])[2:] , doc["id"], bm25) for i, doc in enumerate(query)]
        #tokenized_query = [(re.split(r'\s+', doc["originalText"]), doc["id"], bm25) for i, doc in enumerate(query)]
        bm25_result = []
        time_start=time.time()
        with Pool(cores) as p:
                _result = list((tqdm(p.starmap(search_bm25, tokenized_query), total=len(tokenized_query), desc='bm25')))
                for i in _result:
                    bm25_result.append(i)
        end_start=time.time()
        print('cost time {:.5f} min'.format((end_start - time_start)/ 60 ))
        
        result_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/bm25_result_{}.json'.format(output_type[j])
        with open(result_path, 'w') as f:
            json.dump(bm25_result, f, indent=2)
    
    pool.close()
    pool.join()






    """k = 5
    bm25_result = {}
    for it, i in enumerate(tqdm(tokenized_query)):
        scores = bm25.get_scores(i)
        top_n = np.argsort(scores)[::-1][:k]
        #print(top_n)
        bm25_result[query_url[it]] = [corpus_name[i] for i in top_n]"""
        #print(u)
    #print(bm25_result)
    

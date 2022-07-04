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
from pyserini.search.lucene import LuceneSearcher

def search_bm25(tokenized_query, id, bm25, k=100):
    #tokenized_query = x[0]
    #i_query = x[1]
    #bm25 = x[2]
    result = {}
    hits = searcher.search(tokenized_query, k=100)
    result[id] = {'candidate_doc': [hits[i].docid for i in range(len(hits))], 'score': [hits[i].score for i in range(len(hits))]}
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
    steps = ['2']

    if '1' in steps:
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
                corpus.append({"id": v['file_name'], "contents": v['tapas_set']})
        else:
            assert('file not found error')

        print('all corpus: {}'.format(len(corpus)))


        with jsonlines.open('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tapas_set_document.jsonl', 'w') as f:
            for i in tqdm(range(len(corpus))):
                f.write(corpus[i])

    if '2' in steps:
        cores = 4
        searcher = LuceneSearcher('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tapas_set_index_jsonl')
        
        file_list = ['/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_train.jsonl', '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_dev.jsonl', '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_test.jsonl']
        output_type = ['train', 'dev', 'test']
        for j, file_tdt in enumerate(file_list):
            print(file_tdt)
            query = file2query(file_tdt)
            tokenized_query = [(re.split(r'\s+', doc["question"]) + re.split(r'\s+', doc["question"])[2:] + re.split(r'\s+', doc["question"])[2:] + re.split(r'\s+', doc["question"])[2:] , doc["id"], searcher) for i, doc in enumerate(query)]
            #tokenized_query = [(re.split(r'\s+', doc["originalText"]), doc["id"], bm25) for i, doc in enumerate(query)]
            bm25_result = []
            time_start=time.time()
            for i in tqdm(range(len(query))):
                doc = query[i]
                queryx = re.split(r'\s+', doc["question"]) + re.split(r'\s+', doc["question"])[2:] + re.split(r'\s+', doc["question"])[2:] + re.split(r'\s+', doc["question"])[2:]
                queryx = " ".join(queryx)
                bm25_result.append(search_bm25(queryx, doc["id"], searcher))
           
            end_start=time.time()
            print('cost time {:.5f} min'.format((end_start - time_start)/ 60 ))
            
            result_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/bm25_result_{}.json'.format(output_type[j])
            with open(result_path, 'w') as f:
                json.dump(bm25_result, f, indent=2)


    

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

def file2query(filename):
    query = []
    print('read file...')
    with jsonlines.open(filename, 'r') as f:
        for i in tqdm(f):
            query.append({"question": i["question"], "id": i["id"]})
    print('{} have {} data'.format(filename, str(len(query))))
    return query

if __name__ == '__main__':


    tabfact_file_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process/'   #169898

    
    file_list = ['/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_test.jsonl']
    output_type = ['test']
    for j, file_tdt in enumerate(file_list):
        print(file_tdt)
        table_csv = pd.DataFrame()
        query = file2query(file_tdt)
        for index, i in enumerate(query):
            #table_csv = table_csv.append({'text': i['originalText'], 'answer': i['answer']['answerTexts']}, ignore_index=True)
            table_csv = table_csv.append({'id': str(index),'text': i['question']}, ignore_index=True)
        
        result_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/queries-{}.tsv'.format(output_type[j])
        print(table_csv.iloc[0])
        table_csv.columns = table_csv.iloc[0]
        table_csv = table_csv.drop([0])
        table_csv.to_csv(result_path, sep='\t', index=False)





    """k = 5
    bm25_result = {}
    for it, i in enumerate(tqdm(tokenized_query)):
        scores = bm25.get_scores(i)
        top_n = np.argsort(scores)[::-1][:k]
        #print(top_n)
        bm25_result[query_url[it]] = [corpus_name[i] for i in top_n]"""
        #print(u)
    #print(bm25_result)
    

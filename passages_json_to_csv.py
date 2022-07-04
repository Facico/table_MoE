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

def file2query(filename):
    query = []
    print('read file...')
    with jsonlines.open(filename, 'r') as f:
        for i in tqdm(f):
            query.append(i["questions"][0])
    print('{} have {} data'.format(filename, str(len(query))))
    return query

if __name__ == '__main__':

    json_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/all_raw_table_chunks_for_index.json'   #4473676
    result_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/all_raw_table_chunks.tsv'
    csv.field_size_limit(500 * 1024 * 1024)
    with open(json_path, 'r') as f:
        data = json.load(f)
    num = 0
    f_csv = csv.DictWriter(open(result_path, 'w'), fieldnames=['id', 'text', 'title'], delimiter='\t')
    f_csv.writeheader()
    for i in tqdm(range(len(data))):
        num += 1
        f_csv.writerow({'id': str(num), 'text': data[i]['text'], 'title':data[i]['title']})


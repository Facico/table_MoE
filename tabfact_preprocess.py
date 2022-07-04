from __future__ import division
import random
import sys
import io
import os
import logging
import re
import pandas as pd
import os.path as op
from tqdm import tqdm
from collections import Counter, OrderedDict
import argparse
from multiprocessing import Pool
import multiprocessing
import json
import jsonlines

entity_linking_pattern = re.compile('#.*?;-*[0-9]+,(-*[0-9]+)#')
fact_pattern = re.compile('#(.*?);-*[0-9]+,-*[0-9]+#')
unk_pattern = re.compile('#([^#]+);-1,-1#')
TSV_DELIM = "\t"
TBL_DELIM = " ; "

def join_unicode(delim, entries):
    #entries = [_.decode('utf8') for _ in entries]
    return delim.join(entries)

def parse_fact(fact):
    fact = re.sub(unk_pattern, '[UNK]', fact)
    chunks = re.split(fact_pattern, fact)
    output = ' '.join([x.strip() for x in chunks if len(x.strip()) > 0])
    return output

def tabfact_template(tablex, scan = 'horizontal'):
    #print(tablex['file_name'])
    table_header = tablex['header']
    datax = tablex['data']
    n_row, n_column = len(datax), len(datax[0])
    #print(n_row, n_column)
    table_cells = []
    if scan == 'horizontal':
        for i in range(n_row):
            table_cells.append('row {} is :'.format(i + 1))
            this_row = []
            for j in range(n_column):
                this_row.append('{} is {}'.format(table_header[j][0], datax[i][j][0]))
            this_row = join_unicode(TBL_DELIM, this_row)
            table_cells.append(this_row)
            table_cells.append('.')
    elif scan == 'vertical':
        for j in range(n_column):
            table_cells.append('{} are :'.format(table_header[j][0]))
            this_column = []
            for i in range(n_row):
                this_column.append('row {} is {}'.format(i, datax[i][j][0]))
            this_column = join_unicode(TBL_DELIM, this_column)
            table_cells.append(this_column)
            table_cells.append('.')
    else:
        pass

    table_str = ' '.join(table_cells)
    #print(table_str)
    results = {'table_text': table_str}
    for k, v in tablex.items():
        results[k] = v
    return results

def tabfact_template_all(tablex):
    #print(tablex['file_name'])
    table_header = tablex["header"]
    n_row, n_column = len(tablex["rows"]), len(table_header)
    datax = tablex["rows"]
    #print(n_row, n_column)
    results = {}
    table_cells = []
 
    """table_cells.append('title is {}'.format(tablex['doc_title'].strip(" ")))
    
    for i in range(n_row):
        table_cells.append('row {} is :'.format(i + 1))
        this_row = []
        for j in range(n_column):
            this_row.append('{} is {}'.format(table_header[j].strip(" "), datax[i][j].strip(" ")))
        this_row = join_unicode(TBL_DELIM, this_row)
        table_cells.append(this_row)
        table_cells.append('.')

    table_str = ' '.join(table_cells)
    results['table_text_horizontal'] = table_str.lower()"""

    """table_cells = []
    table_cells.append('title is {}'.format(tablex['doc_title'].strip(" ")))
    for j in range(n_column):
        table_cells.append('{} are :'.format(table_header[j].strip(" ")))
        this_column = []
        for i in range(n_row):
            this_column.append('row {} is {}'.format(i, datax[i][j].strip(" ")))
        this_column = join_unicode(TBL_DELIM, this_column)
        table_cells.append(this_column)
        table_cells.append('.')
    table_str = ' '.join(table_cells)
    results['table_text_vertical'] = table_str.lower()"""
    tapas_set = []
    for i in range(n_row):
        for j in range(n_column):
            tapas_set.append(datax[i][j].strip(" "))
    for i in range(60):
        tapas_set.append(tablex['doc_title'].strip(" "))
    for i in range(15):
        for j in range(n_column):
            tapas_set.append(table_header[j].strip(" "))
    
    results['tapas_set'] = " ".join(tapas_set).lower()

    """table_str = tablex['doc_title'].strip(" ") + ","
    simple_line = []
    for j in range(n_column):
        simple_line.append(table_header[j].strip(" "))
    table_str += ",".join(simple_line) + "\n"
    for i in range(n_row):
        simple_line = []
        for j in range(n_column):
            simple_line.append(datax[i][j].strip(" "))
        table_str += ",".join(simple_line) + "\n"

    results['simple_set'] = table_str.lower()"""
    
    table_str = tablex['doc_title'].strip(" ") + ","
    simple_line = []
    for j in range(n_column):
        simple_line.append(table_header[j].strip(" "))
    table_str += ",".join(simple_line) + "\n"
    for i in range(n_row):
        simple_line = []
        for j in range(n_column):
            simple_line.append(table_header[j].strip(" ") + " " + datax[i][j].strip(" "))
        table_str += ",".join(simple_line) + "\n"

    results['mine_set'] = table_str.lower()

    results['documentTitle'] = tablex['doc_title']
    results['tableId'] = tablex['table_id']
    results['documentUrl'] = tablex['doc_url']
    results['file_name'] = tablex['table_id'].split('_')[-1]
    results["ori_table"] = {"columns": table_header, "rows": datax, 'documentTitle': tablex['doc_title']}

    return results
if __name__ == '__main__':
    #table_path = './data/tables_tok'
    #output_path = './data/tabfact_process'

    """
    tapas table
    """
    table_jsonl_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/all_raw_tables.json'
    output_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    table_data = []
    print('get file...')
    with open(table_jsonl_path, 'r') as f:
        table_data = json.load(f)
    
    cores = 6 #multiprocessing.cpu_count()
    #pool = Pool(cores)
    #rs = pool.map(tabfact_template_all, table_data)
    rs = []
    for i in tqdm(range(len(table_data))):
        rs.append(tabfact_template_all(table_data[i]))
    print('processing...')
    file_set = {}
    table_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json'
    table_all = {}
    for i in tqdm(range(len(rs))):
        tablex = rs[i]
        file_name = tablex['file_name']
        if file_name not in file_set:
            file_set[file_name] = 0
        else:
            file_set[file_name] += 1
        file_name = file_name + '_{}'.format(str(file_set[file_name])) + '.json'
        tablex['file_name'] = file_name
        table_all[file_name] = tablex
        file_path = os.path.join(output_path, file_name)
        with open(file_path, 'w') as f:
            json.dump(tablex, f)
    with open(table_path, 'w') as f:
        json.dump(table_all, f)
    #pool.close()
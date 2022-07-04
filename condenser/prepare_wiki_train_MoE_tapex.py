from email.charset import add_alias
import json
import os
from argparse import ArgumentParser

from tevatron.driver.transformers_MoE import AutoTokenizer, PreTrainedTokenizer
from tevatron.driver.transformers_MoE import TapexTokenizer, BartForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import sys

def seq_to_table(text):
    tables = text.strip('\n').split('\n')
    headers = tables[0].split(', ')
    rows = []
    table_data = {}
    column_id = 0
    add_empty_column = 0
    for i in range(1, len(tables)):
        rows.append(tables[i].split(', '))
        add_empty_column=max(add_empty_column, len(rows[-1]))
    for i in range(add_empty_column - len(headers)):
        headers.append('EMPTY')
    change_id = {}
    for header in headers:
        column = []
        for j in range(len(rows)):
            if column_id >= len(rows[j]):
                column.append('')
            else:
                column.append(rows[j][column_id])
        if header in table_data.keys():
            table_data[header+"_{}".format(str(column_id))] = column
            change_id[header+"_{}".format(str(column_id))] = header
        else:
            table_data[header] = column
        column_id += 1
    
    table = pd.DataFrame.from_dict(table_data)
    column_new = []
    for i in table.columns:
        if 'EMPTY' in i:
            column_new.append('EMPTY')
        else:
            column_new.append(i)
    table.columns = column_new
    return table
    
parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=False, default='facebook/bart-base')
parser.add_argument('--minimum-negatives', type=int, required=False, default=1)
parser.add_argument('--negative_name', type=str, required=False, default="hard_negative_ctxs") # negative_ctxs \ hard_negative_ctxs
parser.add_argument('--add_type', type=int, required=False, default=0) # 0 text, 1 table
args = parser.parse_args()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

tokenizer_table = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")

data = json.load(open(args.input))

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

negative_name = args.negative_name
with open(args.output, 'w') as f:
    for idx, item in enumerate(tqdm(data)):
        if len(item[negative_name]) < args.minimum_negatives or len(item['positive_ctxs']) < 1:
            continue
        group = {}
        positives = [pos['title'].lower() + tokenizer.sep_token + pos['text'] for pos in item['positive_ctxs']]
        negatives = [neg['title'].lower() + tokenizer.sep_token + neg['text'] for neg in item[negative_name]]
        query = tokenizer.encode(item['question'], add_special_tokens=False, max_length=256, truncation=True)
        if args.add_type == 0:
            positives = tokenizer(
                positives, add_special_tokens=False, max_length=256, truncation=True, padding=False)['input_ids']
            negatives = tokenizer(
                negatives, add_special_tokens=False, max_length=256, truncation=True, padding=False)['input_ids']
        else:
            positives_tables = [seq_to_table(pos['text']) for pos in item['positive_ctxs']]
            negatives_tables = [seq_to_table(neg['text']) for neg in item[negative_name]]
            positives = tokenizer_table(
                table=positives_tables, add_special_tokens=False, max_length=256, truncation=True, padding=False)['input_ids']
            
            negatives = tokenizer_table(
                table=negatives_tables, add_special_tokens=False, max_length=256, truncation=True, padding=False)['input_ids']
            for i in range(len(positives)):
                positives[i] = tokenizer.encode(item['positive_ctxs'][i]['title'].lower() + tokenizer.sep_token, add_special_tokens=False, max_length=256, truncation=True) + positives[i]
            for i in range(len(negatives)):
                negatives[i] = tokenizer.encode(item[negative_name][i]['title'].lower() + tokenizer.sep_token, add_special_tokens=False, max_length=256, truncation=True) + negatives[i]
        group['query'] = query
        group['positives'] = positives
        group['negatives'] = negatives

        group['positives_types'] = [args.add_type for i in range(len(positives))]
        group['negatives_types'] = [args.add_type for i in range(len(negatives))]

        f.write(json.dumps(group) + '\n')
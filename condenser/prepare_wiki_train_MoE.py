import json
import os
from argparse import ArgumentParser

from tevatron.driver.transformers_MoE import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=False, default='bert-base-uncased')
parser.add_argument('--minimum-negatives', type=int, required=False, default=1)
parser.add_argument('--negative_name', type=str, required=False, default="hard_negative_ctxs") # negative_ctxs \ hard_negative_ctxs
parser.add_argument('--add_type', type=int, required=False, default=0) # 0 text, 1 table
args = parser.parse_args()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

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
        positives = [pos['title'] + tokenizer.sep_token + pos['text'] for pos in item['positive_ctxs']]
        negatives = [neg['title'] + tokenizer.sep_token + neg['text'] for neg in item[negative_name]]

        query = tokenizer.encode(item['question'], add_special_tokens=False, max_length=256, truncation=True)
        positives = tokenizer(
            positives, add_special_tokens=False, max_length=256, truncation=True, padding=False)['input_ids']
        negatives = tokenizer(
            negatives, add_special_tokens=False, max_length=256, truncation=True, padding=False)['input_ids']

        group['query'] = query
        group['positives'] = positives
        group['negatives'] = negatives

        group['positives_types'] = [args.add_type for i in range(len(positives))]
        group['negatives_types'] = [args.add_type for i in range(len(negatives))]
        
        f.write(json.dumps(group) + '\n')

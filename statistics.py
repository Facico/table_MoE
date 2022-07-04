import json
import jsonlines
from tqdm import tqdm

if __name__ == '__main__':
    file_path = "/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/nq_raw_tables/nq-train_raw_table_pos_bm25_neg.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    o = 0
    for i in tqdm(range(len(data))):
        o += 1
    
    print(o)

    qas_file = '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_train.jsonl'
    o = 0
    with jsonlines.open(qas_file, 'r') as f:
        for i in tqdm(f):
            o += 1
    
    print(o)

    json_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/all_raw_table_chunks_for_index.json'   #4473676
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(len(data))

    json_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/all_raw_tables.json'   #2366545
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(len(data))
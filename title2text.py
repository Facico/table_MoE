import json
from tqdm import tqdm

if __name__ == '__main__':
    table_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json'
    bm25_file_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/nq_raw_tables/nq-train_raw_table_pos_bm25_neg.json'

    with open(bm25_file_path, 'r') as f:
        bm25_file = json.load(f)
    
    title2text = {}
    num = 0
    chong = 0
    for i in tqdm(range(len(bm25_file))):
        for pos_x in bm25_file[i]["positive_ctxs"]:
            if pos_x['title'] not in title2text:
                title2text[pos_x['title']] = pos_x["text"]
                num += 1
            else:
                if pos_x["text"] != title2text[pos_x['title']]:
                    chong += 1
        for neg_x in bm25_file[i]["negative_ctxs"]:
            if neg_x['title'] not in title2text:
                title2text[neg_x['title']] = neg_x["text"]
                num += 1
            else:
                if neg_x["text"] != title2text[neg_x['title']]:
                    chong += 1
    
    print(num)
    print(chong)
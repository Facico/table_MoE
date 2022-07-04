import json
import jsonlines
import os
from tqdm import tqdm

if __name__ == '__main__':
    text_path = "/data1/fch123/DPR/downloads/data/psgs_w100"
    table_path = "/data1/fch123/UDT-QA/condenser/UDT_QA/all_verbalized_table_chunks_for_index.json"
    output_path = "/data1/fch123/UDT-QA-data/downloads/data/text_table_docs/"
    data = []
    with jsonlines.open(text_path, 'r') as f:
        print('read text data...')
        for i in tqdm(f):
            data.append({"docid": i["docid"], "text": i["text"], "title": i["title"], "data_type": 0})
    
    table_data = json.load(open(table_path, 'r'))
    
    print('deal table data...')
    num=0
    for i in tqdm(table_data):
        data.append({"docid": 'table_{}'.format(num), "text": i["text"], "title": i["title"], "data_type": 1})
        num += 1
    
    n_doc = len(data)
    print('all data {}'.format(n_doc))

    for i in range(20):
        out_file = os.path.join(output_path, "docs_{}{}.json".format(0 if i < 10 else "",i))
        print("split num {} to file {}".format(i, out_file))
        f_split = open(out_file, 'w')
        for j in tqdm(range((n_doc // 20) * i, min(n_doc, (n_doc // 20) * (i + 1)))):
            json.dump(data[j], f_split)
            f_split.write('\n')


import json
import jsonlines
import os
from tqdm import tqdm

if __name__ == '__main__':
    text_path = "/data1/fch123/DPR/downloads/data/psgs_w100"
    output_path = "/data/fanchenghao/text_docs"
    data = []
    with jsonlines.open(text_path, 'r') as f:
        print('read text data...')
        for i in tqdm(f):
            data.append({"docid": i["docid"], "text": i["text"], "title": i["title"], "data_type": 0})
    
    
    n_doc = len(data)
    print('all data {}'.format(n_doc))

    for i in range(20):
        out_file = os.path.join(output_path, "docs_{}{}.json".format(0 if i < 10 else "",i))
        print("split num {} to file {}".format(i, out_file))
        f_split = open(out_file, 'w')
        for j in tqdm(range((n_doc // 20) * i, min(n_doc, (n_doc // 20) * (i + 1)))):
            json.dump(data[j], f_split)
            f_split.write('\n')


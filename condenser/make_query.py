import json
import csv
from tqdm import tqdm
if __name__ == '__main__':
    id = 0
    json_f = open('/data2/private/fanchenghao/UDT-QA/condenser/nq-dev-queries.json', 'w')
    with open('/data2/private/fanchenghao/DPR/downloads/data/retriever/qas/nq-dev.csv', 'r') as f:
        df = csv.reader(f, delimiter = '\t')

        for row in tqdm(df):
            data_x = {'text_id': id, 'text': row[0]}
            json.dump(data_x, json_f)
            json_f.write('\n')
            id += 1

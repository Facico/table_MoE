import json
from tqdm import tqdm
import random

def add_type(data, data_type='text'):
    new_data = []
    for i in tqdm(range(len(data))):
        datax = data[i]
        if(len(datax["positive_ctxs"]) > 0):
            datax["positive_ctxs"] = datax["positive_ctxs"][:1]
            datax["positive_ctxs"][0]["data_type"] = data_type
        datax["negative_ctxs"] = []
        if(len(datax["hard_negative_ctxs"]) > 0):
            datax["hard_negative_ctxs"] = datax["hard_negative_ctxs"][:1]
            datax["hard_negative_ctxs"][0]["data_type"] = data_type
        new_data.append(datax)
    return new_data
if __name__ == '__main__':
    # text
    type_list = ['train', 'dev', 'test']
    for typex in type_list:
        if typex == 'test':
            break
        json_file = '/data2/private/fanchenghao/DPR/downloads/data/retriever/biencoder-nq-{}.json'.format(typex)
        data = json.load(open(json_file, 'r'))
        new_data_text = add_type(data, data_type="text")

        json_file = '/data2/private/fanchenghao/DPR/downloads/data/retriever/dpr_tapas_dpr_mine_{}.json'.format(typex)
        data = json.load(open(json_file, 'r'))
        new_data_table = add_type(data, data_type="table")

        new_data = new_data_text + new_data_table
        random.shuffle(new_data)

        with open('/data2/private/fanchenghao/DPR/downloads/data/retriever/dpr_MoE_{}.json'.format(typex), 'w') as f:
            json.dump(new_data, f)



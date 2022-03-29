import json
from tqdm import tqdm

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
    json_file = '/data2/private/fanchenghao/DPR/downloads/data/retriever/biencoder-nq-train.json'
    data = json.load(open(json_file, 'r'))
    new_data = add_type(data, data_type="text")
    with open('/data2/private/fanchenghao/DPR/downloads/data/retriever/biencoder-nq-train_MoE.json', 'w') as f:
        json.dump(new_data, f)

    json_file = '/data2/private/fanchenghao/DPR/downloads/data/retriever/dpr_tapas_dpr_mine_train.json'
    data = json.load(open(json_file, 'r'))
    new_data = add_type(data, data_type="table")
    with open('/data2/private/fanchenghao/DPR/downloads/data/retriever/dpr_tapas_dpr_mine_train_MoE.json', 'w') as f:
        json.dump(new_data, f)


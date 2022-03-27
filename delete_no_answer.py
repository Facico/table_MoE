import os
import json
from tqdm import tqdm

def delete_no(file_name):
    file_name_prefix = file_name[:-5]
    file_name_new = file_name_prefix + '_all_have_positive.json'
    print('new: ', file_name_new)
    with open(file_name, 'r') as f:
        data_list = json.load(f)
    data_new = []
    for i in tqdm(range(len(data_list))):
        data_dict = data_list[i]
        #print(len(data_dict['ctxs']))
        pos_num = 0
        neg_num = 0
        for j in data_dict['ctxs']:
            if j['has_answer'] != False:
                pos_num = 1
            if j['has_answer'] == False:
                neg_num = 1
            if(neg_num == 1 and pos_num == 1):
                break
        if neg_num == 1 and pos_num == 1:
            data_new.append(data_dict)
            #print(i)
    with open(file_name_new, 'w') as f:
        json.dump(data_new, f)
if __name__ == '__main__':
    train_file = './outputs/retrieve_train_out.json'
    dev_file = './outputs/retrieve_dev_out.json'
    test_file = './outputs/retrieve_test_out.json'
    file_list = [dev_file, train_file, test_file]
    for file in file_list:
        delete_no(file)
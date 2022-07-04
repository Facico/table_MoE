import json
import os
from tqdm import tqdm
import time
import gzip
from multiprocessing import Pool
import multiprocessing
import re
from functools import partial
import sys
import jsonlines

def get_text(token_text):
    str_list = []
    for i in token_text:
        str_list.append(i['token'])
    return str_list
def get_html_mask(token_text):
    str_list = []
    for i in token_text:
        str_list.append(i['html_token'])
    return str_list

def get_short_from_htmlmask(text_token, mask_text):
    short_answer = []
    now = 0
    now_answer = []
    for i in range(len(text_token)):
        if(mask_text[i] == True):
            if now == 0: 
                continue
            else:
                now = 0
                if(now_answer != []):
                    short_answer.append(now_answer)
                now_answer = []
        else:
            now_answer.append(text_token[i])
            now = 1
    return short_answer
def get_no_html(text_token, mask_text):
    no_html_answer = []
    for i in range(len(text_token)):
        if mask_text[i] == False:
            no_html_answer.append(text_token[i])
    return no_html_answer
def get_answer(table):
    token_text = get_text(table['document_tokens'])
    mask_text = get_html_mask(table['document_tokens'])
    answers = []
    html_mask = []
    for answer_x in table['long_answer_candidates']:
        start_t, end_t = answer_x['start_token'], answer_x['end_token']
        answers.append(token_text[start_t : end_t])
        html_mask.append(mask_text[start_t : end_t])
    return answers, html_mask

def cell_match(cellx, answer):
    for answerx in answer:
        if(answerx.lower() in cellx):
            return True
    return False

def table_match(tablex, answer):
    """header = tablex["columns"]
    n_row, n_column = len(tablex["rows"]), len(header)
    datax = tablex["rows"]
    str_table_list = []
    if(cell_match(tablex["documentTitle"].lower(), answer) == True):
        return True
    for i in range(len(header)):
        if(cell_match(header[i]["text"].lower(), answer) == True):
            return True
    for i in range(n_row):
       for j in range(n_column):
           #print(datax[i]["cells"][j]["text"])
           if(cell_match(datax[i]["cells"][j]["text"].lower(), answer) == True):
                return True"""
    for ansx in answer:
        if(ansx.lower() in tablex.lower()):
            return True
    return False


if __name__ == '__main__':
    cores = 5#multiprocessing.cpu_count()

    step = ['4', '5'] #['1', '2', '4', '5']#['1', '2', '4']
    
    if '1' in step:
        id_to_answer_file = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/id_to_answer_train.json'
        id_to_answer = {}
        train_file = '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_train.jsonl'
        with jsonlines.open(train_file, 'r') as f:
            print('read file ...')
            for i in tqdm(f):
                id, answer = i["id"], i['answers']
                id_to_answer[id] = answer
        with open(id_to_answer_file, 'w') as f:
            json.dump(id_to_answer, f, indent=2)
    
    if '2' in step:
        result_all = []
        all_result_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/bm25_result_train.json'
        with open(all_result_path, 'r') as f:
            one_bm25_result = json.load(f)
            result_all = one_bm25_result
        print('len of all is {}'.format(len(result_all)))
        
        table_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process/'
        print('load table ...')
        table_dict = {}
        if os.path.exists('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json') == True:
            table_tab = json.load(open('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json', 'r'))
            for k, v in table_tab.items():
                table_dict[k] = v["tapas_set"]
        else:
            assert('file not found error')
        print('load finish.')
        #print(table_dict)
        
        start_time = time.time()
        id_to_answer = {}
        id_to_answer_file = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/id_to_answer_train.json'
        with open(id_to_answer_file, 'r') as f:
            id_to_answer = json.load(f)

        ans_pn = {}
        hyper_answer = {}
        pos_all = 0
        for i in tqdm(range(len(result_all))):
            example_id = ''
            for k, v in result_all[i].items():
                example_id = k
            
            if(example_id not in id_to_answer):
                print(example_id)
                continue
            
            answers = id_to_answer[example_id]
            posi_list = []
            positive_num = 0
            for answer_doc_file_name in result_all[i][example_id]["candidate_doc"]:
                tablex = table_dict[answer_doc_file_name]
                y_n = 0
                
                if table_match(tablex, answers):
                    y_n = 1
                    positive_num += 1
                posi_list.append(y_n)
            if positive_num > 0:
                pos_all += 1
            
            #if example_id in ans_pn:
            #    print(example_id, 'youle')
            ans_pn[example_id] = {'pn_list': posi_list}
        print(pos_all, len(result_all))
        end_start = time.time()
        print('cost time {:.5f} min'.format((end_start - start_time)/ 60 ))
        pn_file_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/bm25_pn_train.json'
        with open(pn_file_path, 'w') as f:
            json.dump(ans_pn, f)
    
    if '4' in step:
        ans_pn_all = {}
        pn_all_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/bm25_pn_train.json'  
        with open(pn_all_path, 'r') as f:
            ans_pn_all = json.load(f)
        AP_id = [1, 5, 10, 20, 50, 100]
        for j in AP_id:
            acc_j = 0
            all_j = 0
            for k, v in ans_pn_all.items():
                acc = 0
                for i in range(j):
                    if v['pn_list'][i] == 1:
                        acc += 1
                acc_j += acc / j
                all_j += 1
            print('p@{}: {:.3f}'.format(j, acc_j / all_j))

        acc = 0
        all_j = 0
        tot = 0
        for k, v in ans_pn_all.items():
            acc = 0
            tot += 1
            for i in range(len(v['pn_list'])):
                if v['pn_list'][i] == 1:
                    acc += 1
            if(acc > 0):
                all_j += 1
                #print(k, v)
        print('have positive {} / {} : {}'.format(all_j, tot, all_j / tot))

    if '5' in step:
        ans_pn_all = {}
        pn_all_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/bm25_pn_train.json'  
        with open(pn_all_path, 'r') as f:
            ans_pn_all = json.load(f)
        bm25_result_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/bm25_result_train.json'
        with open(bm25_result_path, 'r') as f:
            bm25_result = json.load(f)
        id_to_table_dict_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/table/id_to_table.json'
        with open(id_to_table_dict_path, 'r') as f:
            id_to_table_dict = json.load(f)
        train_jsonl_path = '/data2/private/fanchenghao/UDT-QA/downloads/data/retriever/qas/nq/nq_table_answerable_train.jsonl'
        positive_one = {}
        with jsonlines.open(train_jsonl_path, 'r') as f:
            print('read train file ...')
            for i in tqdm(f):
                positive_one[i['id']] = [i["table"]["table_id"]] + i["table"]["alternative_table_ids"]
                
        
        AP_id = [1, 5, 10, 20, 50, 100]
        for j in AP_id:
            r = 0
            tot = 0
            for i in range(len(bm25_result)):
                example_id = ''
                for k, v in bm25_result[i].items():
                    example_id = k
                query_id = example_id #'_'.join(example_id.split('_')[:-1])
                positive_list = []
                for o in positive_one[query_id]:
                    if o not in id_to_table_dict:
                        #print('warning {} not in table set'.format(o))
                        continue
                    positive_list.append(id_to_table_dict[o])
                
                yes = 0
                for k in range(j):
                    answer_doc_file_name = bm25_result[i][example_id]["candidate_doc"][k]
                    if answer_doc_file_name in positive_list:
                        yes += 1
                
                r += yes / 1 #yes / len(positive_list)
                tot += 1
            print('R@{}: {:.3f}'.format(j, r / tot))
        

    
    
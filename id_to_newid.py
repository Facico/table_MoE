import json
import os
from tqdm import tqdm

if __name__ == '__main__':
    print('load table ...')
    if os.path.exists('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json') == True:
        table_data = json.load(open('/data2/private/fanchenghao/UDT-QA/downloads/data/table/tabfact_process_all.json', 'r'))
    tot = 0
    table_dict = {}
    for k, v in table_data.items():
        if v["tableId"] not in table_dict:
            tot += 1
            table_dict[v["tableId"]] = k
    with open('/data2/private/fanchenghao/UDT-QA/downloads/data/table/id_to_table.json', 'w') as f:
        json.dump(table_dict, f, indent=2)
    print('all {}'.format(tot))

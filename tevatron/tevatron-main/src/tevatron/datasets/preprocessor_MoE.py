import pandas as pd
class TrainPreProcessor_MoE:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        return {'query': query, 'positives': positives, 'negatives': negatives}


class QueryPreProcessor_MoE:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor_MoE:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text, 'data_type': example['data_type']}


def seq_to_table(text):
    tables = text.strip('\n').split('\n')
    headers = tables[0].split(', ')
    rows = []
    table_data = {}
    column_id = 0
    add_empty_column = 0
    for i in range(1, len(tables)):
        rows.append(tables[i].split(', '))
        add_empty_column=max(add_empty_column, len(rows[-1]))
    for i in range(add_empty_column - len(headers)):
        headers.append('EMPTY')
    change_id = {}
    for header in headers:
        column = []
        for j in range(len(rows)):
            if column_id >= len(rows[j]):
                column.append('')
            else:
                column.append(rows[j][column_id])
        if header in table_data.keys():
            table_data[header+"_{}".format(str(column_id))] = column
            change_id[header+"_{}".format(str(column_id))] = header
        else:
            table_data[header] = column
        column_id += 1
    
    table = pd.DataFrame.from_dict(table_data)
    column_new = []
    for i in table.columns:
        if 'EMPTY' in i:
            column_new.append('EMPTY')
        else:
            column_new.append(i)
    table.columns = column_new
    table = table.astype(str) 
    return table

class CorpusPreProcessor_tapas_MoE:
    def __init__(self, tokenizer_text, tokenizer_table, text_max_length=256, separator_text=' ', separator_table=' '):
        self.tokenizer_text = tokenizer_text
        self.tokenizer_table = tokenizer_table
        self.text_max_length = text_max_length
        self.separator_text = separator_text
        self.separator_table = separator_table
    def __call__(self, example):
        """self.num += 1
        if self.num % 20000 == 0:
            print("yyy"+str(self.num)+"yyy")
        if self.num <= 150000:
            return None"""
        if example['data_type'] == 0:
            docid = example['docid']
            text = example['title'] + self.separator_text + example['text'] if 'title' in example else example['text']
            text = self.tokenizer_text.encode(text,
                                        add_special_tokens=False,
                                        max_length=self.text_max_length,
                                        truncation=True)
        else:
            docid = example['docid']
            try:
                text = self.tokenizer_table(table=seq_to_table(example['text']), 
                                            add_special_tokens=False, 
                                            max_length=self.text_max_length, 
                                            truncation=True)['input_ids']
            except:
                text = self.tokenizer_text.encode(example['text'],
                                        add_special_tokens=False,
                                        max_length=self.text_max_length,
                                        truncation=True)
            text = self.tokenizer_text.encode(example['title'].lower() + self.separator_table, 
                                    add_special_tokens=False, 
                                    max_length=256, truncation=True) + text
        return {'text_id': docid, 'text': text, 'data_type': example['data_type']}
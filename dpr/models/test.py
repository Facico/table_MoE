from transformers import BertConfig, BertModel
from transformers import AdamW
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from copy import deepcopy

if __name__ == '__main__':
    model = BertModel.from_pretrained('bert-base-uncased')
    model_dict_copy = {}
    for k, v in model.state_dict().items():
        if 'bias' in k:
            bias_only = ".".join(k.split('.')[:-1]) + '.bias'
            model_dict_copy[k] =  model.state_dict()[bias_only]
        else:
            model_dict_copy[k] =  model.state_dict()[k]

    model.load_state_dict(model_dict_copy)
    
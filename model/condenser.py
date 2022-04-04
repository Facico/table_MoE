from transformers import AutoModel
from transformers.models.bert.modeling_bert_MoE import MoE_BertModel
model = MoE_BertModel.from_pretrained('Luyu/co-condenser-wiki')

if __name__ == '__main__':
    for k, v in model.state_dict().items():
        if 'bias' in k:
            print(k)
import torch
from tevatron.driver.transformers_MoE  import TapasTokenizer, BertTokenizer, TapasModel
import pandas as pd

# initialize the tokenizer
model = TapasModel.from_pretrained("google/tapas-base-finetuned-wtq")
"""tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data).astype(str) 
tables = [table, table]
query = "select year where city = beijing"


encoding = tokenizer(table=tables,
                     truncation=True, padding="max_length", return_tensors="pt")
print(encoding.keys())


print(tokenizer_text.decode(encoding["input_ids"][0]))"""



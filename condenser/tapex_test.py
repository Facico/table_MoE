from tevatron.driver.transformers_MoE import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wtq")

data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)
tables = [table, table]
query = [None, None]
print(table)
# tapex accepts uncased input since it is pre-trained on the uncased corpus
#query = "select year where city = beijing"
encoding = tokenizer(table=tables, return_tensors="pt")
print(encoding['input_ids'].shape)
outputs = model.generate(**encoding)

print(outputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# ['2008']

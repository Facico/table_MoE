import torch
from tevatron.driver.transformers_MoE  import TapasTokenizer

# initialize the tokenizer
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")



encoding = tokenizer(table=table, queries=item.question, answer_coordinates=item.answer_coordinates, answer_text=item.answer_text,
                     truncation=True, padding="max_length", return_tensors="pt")
encoding.keys()


tokenizer.decode(encoding["input_ids"][0])



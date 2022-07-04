/home/fanchenghao/miniconda3/envs/py37/bin/python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /data2/private/fanchenghao/UDT-QA/downloads/data/table/jsonl \
  --index /data2/private/fanchenghao/UDT-QA/downloads/data/table/tapas_set_index_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 2 \
  --storePositions --storeDocvectors --storeRaw
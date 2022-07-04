#/home/fanchenghao/miniconda3/envs/py37/bin/python -m pyserini.search.lucene \
#  --index /data2/private/fanchenghao/UDT-QA/downloads/data/table/tapas_set_index_jsonl \
#  --topics /data2/private/fanchenghao/UDT-QA/downloads/data/table/queries-test.tsv \
#  --output /data2/private/fanchenghao/UDT-QA/downloads/data/table/run_test.txt \
#  --bm25


/home/fanchenghao/miniconda3/envs/py37/bin/python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
  --index /data2/private/fanchenghao/UDT-QA/downloads/data/table/tapas_set_index_jsonl \
  --topics /data2/private/fanchenghao/UDT-QA/downloads/data/table/queries-test.tsv \
  --input /data2/private/fanchenghao/UDT-QA/downloads/data/table/run_test.txt \
  --output /data2/private/fanchenghao/UDT-QA/downloads/data/table/run_test.json


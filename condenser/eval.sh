RUN="/data1/fch123/UDT-QA/condenser/run.nq.bert.text.table.raw.txt"
trec_out="/data1/fch123/UDT-QA/condenser/run.nq.text.table.teIn"
json_out="/data1/fch123/UDT-QA/condenser/run.nq.test.json"
/home/fanchenghao/miniconda3/envs/py37/bin/python -m tevatron.utils.format.convert_result_to_trec \
    --input $RUN --output $trec_out


/home/fanchenghao/miniconda3/envs/py37/bin/python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run --topics dpr-nq-test \
                                                                --index wikipedia-dpr \
                                                                --input $trec_out \
                                                                --output $json_out

/home/fanchenghao/miniconda3/envs/py37/bin/python -m pyserini.eval.evaluate_dpr_retrieval --retrieval $json_out \
    --topk 5 20 100
ENCODE_QRY_DIR="/data2/private/fanchenghao/UDT-QA/condenser/embeddings-nq-queries/"
ENCODE_DIR="/data2/private/fanchenghao/UDT-QA/condenser/embeddings-nq-dpr/"
DEPTH=100
RUN="/data2/private/fanchenghao/UDT-QA/condenser/run.nq.cc.text.table.txt"
OUTDIR="./temp"
wiki_dir="/data2/private/fanchenghao/UDT-QA/condenser/wikipedia-corpus"

INTERMEDIATE_DIR="/data2/private/fanchenghao/UDT-QA/condenser/intermediate"

for s in $(seq -f "%02g" 0 19)
do

echo $s

/home/fanchenghao/miniconda3/envs/py37/bin/python -m tevatron.faiss_retriever \ 
--query_reps $ENCODE_QRY_DIR/query.pt \
--passage_reps $ENCODE_DIR/${s}.pt \
--depth $DEPTH \
--save_ranking_to ${INTERMEDIATE_DIR}/${s}

done

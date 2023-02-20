ENCODE_QRY_DIR="/data1/fch123/UDT-QA/condenser/embeddings-nq-position-queries-raw/"
ENCODE_DIR="/data1/fch123/UDT-QA/condenser/embeddings-nq-position-raw"
DEPTH=100
RUN="/data1/fch123/UDT-QA/condenser/run.nq.cc.text.table.txt"
OUTDIR="./temp"
wiki_dir="/data1/fch123/UDT-QA/condenser/wikipedia-corpus" #"/data2/private/fanchenghao/DPR/downloads/psgs_w100.tsv"
INTERMEDIATE_DIR="/data1/fch123/UDT-QA/condenser/intermediate"
MODEL_DIR=nq-model
for s in $(seq -f "%02g" $1 $2)
do
echo $s
/home/fanchenghao/miniconda3/envs/py37/bin/python -m tevatron.faiss_retriever \
--query_reps $ENCODE_QRY_DIR/query.pt \
--passage_reps $ENCODE_DIR/$s.pt \
--depth $DEPTH \
--save_ranking_to ${INTERMEDIATE_DIR}/${s}
done

#--batch_size 32 \
#--save_text \
#--save_ranking_to $RUN
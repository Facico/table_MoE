INTERMEDIATE_DIR="/data1/fch123/UDT-QA/condenser/intermediate"
ENCODE_QRY_DIR="/data1/fch123/UDT-QA/condenser/embeddings-nq-queries/"
RUN="/data1/fch123/UDT-QA/condenser/run.nq.bert.text.table.raw.txt"
python -m tevatron.faiss_retriever.reducer \
--score_dir ${INTERMEDIATE_DIR} \
--query $ENCODE_QRY_DIR/query.pt \
--save_ranking_to $RUN
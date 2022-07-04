output_dir="/data1/fch123/UDT-QA/condenser/nq-train-raw-table-tapas"
nq_train_path="/data1/fch123/DPR/downloads/data/retriever/nq-train.json"
output_path="${output_dir}/bm25.bert.json"
hn_path="/data1/fch123/UDT-QA/condenser/biencoder-nq-adv-hn-train.json"
output_hn_path="${output_dir}/hn.bert.json"

#table_raw
table_bm25="/data1/fch123/UDT-QA/condenser/UDT_QA/nq-train_raw_table_pos_bm25_neg.json"
table_dpr="/data1/fch123/UDT-QA/condenser/UDT_QA/nq-train_raw_table_pos_dpr_neg.json"

output_table_bm25="${output_dir}/bm25.table.json"
output_table_dpr="${output_dir}/hn.table.json"

python prepare_wiki_train_MoE_tapas.py --input $nq_train_path --output $output_path

python prepare_wiki_train_MoE_tapas.py --input $hn_path --output $output_hn_path

python prepare_wiki_train_MoE_tapas.py --input $table_bm25 --output $output_table_bm25 --negative_name negative_ctxs --add_type 1

python prepare_wiki_train_MoE_tapas.py --input $table_dpr --output $output_table_dpr --negative_name negative_ctxs --add_type 1




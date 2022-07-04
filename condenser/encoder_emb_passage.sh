OUTDIR="./temp"
wiki_dir="/data/fanchenghao/text_docs" #"/data1/fch123/UDT-QA-data/downloads/data/text_raw_table_docs" #"/data2/private/fanchenghao/DPR/downloads/psgs_w100.tsv"
model_path="/data1/fch123/UDT-QA/condenser/model_bart_text/" #"facebook/dpr-ctx_encoder-single-nq-base" #$CONDENSER_MODEL_NAME #"/data2/private/fanchenghao/UDT-QA/condenser/model_nq3/"
cache_path="/data1/fch123/UDT-QA/condenser/.cache/"
emb_nq_path="/data1/fch123/UDT-QA/condenser/embeddings-nq"
emb_query_path="/data1/fch123/UDT-QA/condenser/embeddings-nq-queries/"
query_path="/data1/fch123/UDT-QA/condenser/nq-test-queries.json"
MODEL_DIR=nq-model

echo $1
for s in $(seq -f "%02g" $2 $3)
do
echo $s
CUDA_VISIBLE_DEVICES=$1 /home/fanchenghao/miniconda3/envs/py37/bin/python -m tevatron.driver.encode \
  --output_dir=$OUTDIR \
  --cache_dir $cache_path \
  --model_name_or_path $model_path/passage_model \
  --tokenizer_name $model_path \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --p_max_len 256 \
  --dataset_proc_num 8 \
  --dataset_name json_not_encode \
  --encode_in_path $wiki_dir/docs_$s.json \
  --encoded_save_path $emb_nq_path/$s.pt \
  --passage_field_separator sep_token \
  --data_cache_dir $cache_path \
  --dataloader_num_workers 4
done

# --encode_in_path $wiki_dir/docs$s.json \
#
  

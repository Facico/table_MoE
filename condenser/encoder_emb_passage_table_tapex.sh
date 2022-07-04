OUTDIR="./temp"
wiki_dir="/data1/fch123/UDT-QA-data/downloads/data/text_raw_table_docs" #"/data2/private/fanchenghao/DPR/downloads/psgs_w100.tsv"
model_path="/data1/fch123/UDT-QA//condenser/model_MoE_tapex/" #"model_bert_prompt_raw" #model_bert_position_prompt_raw" #"text_raw_table_bert" #"/data1/fch123//UDT-QA/condenser/model_bert_text_table_raw/" #"facebook/dpr-ctx_encoder-single-nq-base" #$CONDENSER_MODEL_NAME #"/data2/private/fanchenghao/UDT-QA/condenser/model_nq3/"
emb_nq_path="/data1/fch123/UDT-QA/condenser/embeddings-nq/" #embeddings-nq-dpr
emb_query_path="/data1/fch123/UDT-QA/condenser/embeddings-nq-queries/"
query_path="/data1/fch123/UDT-QA/condenser/nq-test-queries.json"
#cache_path="/data2/private/fanchenghao/UDT-QA/condenser/.cache/"
cache_path="/data1/fch123/.cache" #MoE


echo $1
for s in $(seq -f "%02g" $2 $3)
do
echo $s
CUDA_VISIBLE_DEVICES=$1 python -m tevatron.driver.encode_MoE_tapex \
  --output_dir=$OUTDIR \
  --cache_dir $cache_path \
  --model_name_or_path $model_path/passage_model \
  --table_model_name_or_path $model_path/table_model \
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

#--soft_prompt \
#  --prompt_tokens 20 \
#  --initialize_from_vocab

# --encode_in_path $wiki_dir/docs$s.json \
# #json_not_encode Tevatron/wikipedia-nq-corpus\

#--encode_num_shard 20 \
#--encode_shard_index $s \
  

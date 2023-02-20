OUTDIR="./temp"
CONDENSER_MODEL_NAME="/data1/fch123/UDT-QA/condenser/model_MoE_position_raw_table_3/" #"/data1/fch123/UDT-QA/condenser/model_bert_prompt_raw_2//" #"/data1/fch123/UDT-QA/condenser/model_bert_text_table_raw/" #"facebook/dpr-question_encoder-single-nq-base" #"/data2/private/fanchenghao/UDT-QA/ANCE/model/co-condenser-wiki"
model_path=$CONDENSER_MODEL_NAME #"/data2/private/fanchenghao/UDT-QA/condenser/model_nq3/"
cache_path="/data1/fch123/UDT-QA/condenser/.cache/"
emb_nq_path="/data1/fch123/UDT-QA/condenser/embeddings-nq-position-raw" #embeddings-nq-dpr
emb_query_path="/data1/fch123/UDT-QA/condenser/embeddings-nq-position-queries-raw/"
query_path="/data1/fch123/UDT-QA/condenser/nq-test-queries.json"
cache_path="/data1/fch123/UDT-QA/condenser/.cache/"
MODEL_DIR=nq-model


# query

CUDA_VISIBLE_DEVICES=$1 python -m tevatron.driver.encode_position_MoE \
  --output_dir=$OUTDIR \
  --model_name_or_path $model_path/query_model\
  --tokenizer_name $model_path \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --dataset_proc_num 2 \
  --encode_in_path $query_path \
  --encoded_save_path $emb_query_path/query.pt \
  --encode_is_qry
  
  #--soft_prompt \
  #--prompt_tokens 1 \
  #--initialize_from_vocab
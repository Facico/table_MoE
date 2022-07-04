OUTDIR="./temp"
CONDENSER_MODEL_NAME="/data1/fch123/UDT-QA//condenser/model_MoE_tapas/" #"/data1/fch123/UDT-QA/condenser/model_bert_prompt_raw_2//" #"/data1/fch123/UDT-QA/condenser/model_bert_text_table_raw/" #"facebook/dpr-question_encoder-single-nq-base" #"/data2/private/fanchenghao/UDT-QA/ANCE/model/co-condenser-wiki"
model_path=$CONDENSER_MODEL_NAME #"/data2/private/fanchenghao/UDT-QA/condenser/model_nq3/"
cache_path="/data1/fch123/.cache"
emb_nq_path="/data1/fch123/UDT-QA/condenser/embeddings-nq" #embeddings-nq-dpr
emb_query_path="/data1/fch123/UDT-QA/condenser/embeddings-nq-queries/"
query_path="/data1/fch123/UDT-QA/condenser/nq-test-queries.json"
cache_path="/data1/fch123/.cache"
MODEL_DIR=nq-model


# query

CUDA_VISIBLE_DEVICES=$1 /home/fanchenghao/miniconda3/envs/py37/bin/python -m tevatron.driver.encode_MoE_tapas \
  --output_dir=$OUTDIR \
  --model_name_or_path $model_path/query_model\
  --table_model_name_or_path $model_path/table_model \
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
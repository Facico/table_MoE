CONDENSER_MODEL_NAME="bert-base-uncased" #"/data2/private/fanchenghao/UDT-QA/ANCE/model/co-condenser-wiki" #"bert-base-uncased" #"/data2/private/fanchenghao/UDT-QA/ANCE/model/co-condenser-wiki"
#CONDENSER_MODEL_NAME="data2/private/fanchenghao/UDT-QA/condenser/model_nq/"
table_model_path="google/tapas-base-finetuned-wtq"
train_path="/data1/fch123/UDT-QA/condenser/nq-train-raw-table-tapas"
output_path="/data1/fch123/UDT-QA/condenser/model_MoE_tapas/"
cache_path="/data/fanchenghao/.cache"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.MoE_train_tapas \
  --output_dir $output_path \
  --model_name_or_path $CONDENSER_MODEL_NAME \
  --table_model_name_or_path $table_model_path \
  --cache_dir $cache_path \
  --do_train \
  --save_steps 5000 \
  --train_dir $train_path \
  --fp16 \
  --per_device_train_batch_size 128 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 40 \
  --negatives_x_device \
  --positive_passage_no_shuffle \
  --untie_encoder \
  --grad_cache \
  --gc_p_chunk_size 8 \
  --gc_q_chunk_size 8 \
  --gc_type_chunk_size 8

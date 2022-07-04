CONDENSER_MODEL_NAME="/data2/private/fanchenghao/UDT-QA/condenser/model_bert_text_table/"  #"/data2/private/fanchenghao/UDT-QA/ANCE/model/co-condenser-wiki" #"bert-base-uncased" #"/data2/private/fanchenghao/UDT-QA/ANCE/model/co-condenser-wiki"
#CONDENSER_MODEL_NAME="data2/private/fanchenghao/UDT-QA/condenser/model_nq/"
train_path="/data2/private/fanchenghao/nq-train-MoE"
output_path="/data2/private/fanchenghao/UDT-QA/condenser/model_bert_bias/"
cache_path="/data3/private/fanchenghao/.cache"
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train_bitfit \
  --output_dir $output_path \
  --model_name_or_path $CONDENSER_MODEL_NAME \
  --cache_dir $cache_path \
  --do_train \
  --save_steps 10000 \
  --train_dir $train_path \
  --fp16 \
  --per_device_train_batch_size 128 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 1 \
  --negatives_x_device \
  --positive_passage_no_shuffle \
  --untie_encoder \
  --grad_cache \
  --gc_p_chunk_size 16 \
  --gc_q_chunk_size 8


gpu_no=2
#encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-07/11-54-47/outputs/dpr_biencoder.39" #template
#encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-18/06-36-58/outputs/dpr_biencoder.35" #simple
#encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-22/10-46-59/outputs/dpr_biencoder.38"
#encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-23/14-37-18/outputs/dpr_biencoder.34"
output_file="/data2/private/fanchenghao/DPR/outputs/embedding/table_emb"
#ctx_path="/data2/private/fanchenghao/DPR/downloads/data/retriever/table_simple_list.tsv"
#python generate_dense_embeddings.py \
#	model_file=$encoder_checkpoint_path \
#	ctx_src=dpr_table
#	out_file=$output_file

for i in {3..4}; do
  echo $((i-3))
  export CUDA_VISIBLE_DEVICES=${i}
  python generate_dense_embeddings.py  \
    model_file=$encoder_checkpoint_path   \
    ctx_src=dpr_wiki \
    shard_id=$((i-3)) \
    num_shards=2 \
    out_file=$output_file
done
ngpu=2
#encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-07/11-54-47/outputs/dpr_biencoder.39"
#encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-18/06-36-58/outputs/dpr_biencoder.35"
#encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-22/10-46-59/outputs/dpr_biencoder.38"
encoder_checkpoint_path="/data2/private/fanchenghao/DPR/outputs/2022-03-23/14-37-18/outputs/dpr_biencoder.34"
#CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=$ngpu dense_retriever.py \
#	model_file=$encoder_checkpoint_path \
#	qa_dataset=nq_test  \
#	ctx_datatsets=[dpr_table] \
#	encoded_ctx_files=[\"/data3/private/fanchenghao/DPR/outputs/embedding/table_emb*\"] \
#	out_file="./outputs/retrieve_out.json"

CUDA_VISIBLE_DEVICES=3
python dense_retriever.py \
	model_file=$encoder_checkpoint_path \
	qa_dataset=tapas_test  \
	ctx_datatsets=[mine_dpr_table] \
    encoded_ctx_files=[\"/data2/private/fanchenghao/DPR/outputs/embedding/table_emb*\"] \
	out_file="/data2/private/fanchenghao/DPR/outputs/retrieve_train_out.json"
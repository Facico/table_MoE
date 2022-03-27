ngpu=4
encoder_checkpoint_path="/data3/private/fanchenghao/DPR/outputs/2022-02-08/21-30-08/outputs/dpr_biencoder.39"
train_file="/data3/private/fanchenghao/DPR/outputs/retrieve_train_out_all_have_positive.json"
dev_file="/data3/private/fanchenghao/DPR/outputs/retrieve_dev_out_all_have_positive.json"

#pre-processing
#CUDA_VISIBLE_DEVICES=1 python train_extractive_reader.py \
#	encoder.sequence_length=350 \
#	train_files="$train_file" \
#	dev_files="$dev_file"  \
#	output_dir="./outputs/"

CUDA_VISIBLE_DEVICES=1,2,4,6 python train_extractive_reader.py \
	encoder.sequence_length=270 \
	train_files="$train_file" \
	dev_files="$dev_file"  \
	output_dir="./outputs/"
gpu_no=2
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_port=11451 train_dense_encoder.py \
train_datasets=[dpr_MoE_train] \
dev_datasets=[dpr_MoE_dev] \
train=biencoder_local \
output_dir="./outputs"
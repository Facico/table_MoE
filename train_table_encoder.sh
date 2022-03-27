gpu_no=2
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=$gpu_no train_dense_encoder.py \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir="./outputs"
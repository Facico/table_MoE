checkpoint_path="/data3/private/fanchenghao/DPR/outputs/2022-02-08/21-30-08/outputs"
for ((i=0;i<=35;i++)); do
echo ${checkpoint_path}/dpr_biencoder.$i
rm ${checkpoint_path}/dpr_biencoder.$i
done
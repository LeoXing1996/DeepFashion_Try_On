export CUDA_VISIBLE_DEVICES="0,1,2,3"

python train.py --batchSize 8 --nThreads 8 --gpu_ids 0,1,2,3 --continue_train \
                --load_pretrain checkpoints/label2city \
                --which_epoch latest

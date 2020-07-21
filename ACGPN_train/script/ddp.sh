export CUDA_VISIBLE_DEVICES="4,5"

cd ../

python -m torch.distributed.launch --nproc_per_node=2 train.py \
                --name ddp_test --batchSize 4 --nThreads 16 --gpu_ids 0,1 \
                --niter 40 --niter_decay 0
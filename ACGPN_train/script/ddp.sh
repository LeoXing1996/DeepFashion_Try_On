export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd ../
python -m torch.distributed.launch --nproc_per_node=4 \
       train.py --batchSize 8 --nThreads 16 --gpu_ids 0,1,2,3 \
                --name DDP_Baseline \
                --niter 40 --niter_decay 0
export CUDA_VISIBLE_DEVICES="1"

python train.py --name noWarpingNoMask --batchSize 2 --nThreads 16 --gpu_ids 0 \
                --niter 40 --niter_decay 0

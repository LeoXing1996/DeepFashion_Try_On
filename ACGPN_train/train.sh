export CUDA_VISIBLE_DEVICES="0,1,2,3"

python train.py --name noWarping --batchSize 8 --nThreads 16 --gpu_ids 0,1,2,3 \
                --niter 40 --niter_decay 0

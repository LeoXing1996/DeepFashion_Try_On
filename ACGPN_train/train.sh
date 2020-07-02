export CUDA_VISIBLE_DEVICES="0,1,2,3"

python train.py --name inPaintingE2E --batchSize 8 --nThreads 8 --gpu_ids 0,1,2,3 \
                --niter 30 --niter_decay 0

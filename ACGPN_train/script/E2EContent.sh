export CUDA_VISIBLE_DEVICES="1"

# the following setting remove L2 loss of warped mask
# python train.py --name predictedMaskL2_onlyL2Cloth --batchSize 2 --nThreads 16 --gpu_ids 0 \
#                 --niter 40 --niter_decay 0

# the following setting add end2end training of content fusion module
python train.py --name E2Econtent --batchSize 2 --nThreads 16 --gpu_ids 0 \
                --niter 40 --niter_decay 0
export CUDA_VISIBLE_DEVICES=0
for ep in {10..200..10}
do
    # echo ${ep} start
    # name=EP${ep}_eval_rec
    # python test.py --load_pretrain ../ACGPN_train/checkpoints/label2city \
    #                --which_epoch ${ep} --name ${name}
    # python run_ssim.py --name ${name}
    file_path=./sample/EP${ep}_eval_rec/SSIM.txt
    SSIM=$(cat ${file_path})
    echo ${ep}-${SSIM}
done

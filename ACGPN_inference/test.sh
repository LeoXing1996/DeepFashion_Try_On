export CUDA_VISIBLE_DEVICES=1

SETTING='noWarping'

for ep in {10..40..10}
do
    name=EP${ep}_eval_rec
    python test.py --which_ckpt ${SETTING} --which_epoch ${ep} --name ${name} --debug
    python run_ssim.py --which_ckpt ${SETTING} --name ${name}
done

for ep in {10..40..10}
do
    file_path=./sample/${SETTING}/EP${ep}_eval_rec/SSIM.txt
    SSIM=$(cat ${file_path})
    echo ${ep}-${SSIM}
done

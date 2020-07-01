export CUDA_VISIBLE_DEVICES=5

python test.py --load_pretrain ../ACGPN_train/checkpoints/offical_release --which_epoch latest --name debug --debug
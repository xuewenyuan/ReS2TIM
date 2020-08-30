#! /bin/bash
python train.py --dataroot ./datasets/icdar13table --gpu_ids 7 --model res2tim --dataset_mode cell_rel --lr 0.0005 --pair_batch 10000 --niter 5 --niter_decay 95 --use_mask --name delet --continue_train --epoch prt

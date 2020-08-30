#! /bin/bash
python test.py --dataroot ./datasets/icdar13table --gpu_ids 7 --model res2tim --dataset_mode cell_rel --pair_batch 10000 --use_mask --name res2tim_icdar13table --epoch best

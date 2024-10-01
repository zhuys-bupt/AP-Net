#!/usr/bin/env bash
set -x
DATAPATH="your"
python main.py  --dataset eth3d \
                --datapath $DATAPATH \
                --trainlist ./filenames/eth3d_all.txt \
                --testlist ./filenames/eth3d_all.txt \
                --batch_size 4 \
                --test_batch_size 1 \
                --train_num_workers 4 \
                --test_num_workers 1 \
                --epochs 300 \
                --lrepochs "200:10" \
                --model gwcnet-ca \
                --logdir your \
                --loadckpt your \


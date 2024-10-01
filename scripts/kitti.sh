#!/usr/bin/env bash
set -x
DATAPATH="your"
python main.py  --dataset kitti \
                --datapath $DATAPATH \
                --trainlist ./filenames/kitti12_15_all.txt \
                --testlist ./filenames/kitti12_15_all.txt \
                --epochs 600 \
                --lrepochs "400:10" \
                --model gwcnet-ca \
                --logdir your \
                --loadckpt your \











#!/usr/bin/env bash
set -x
DATAPATH="your"
python validation.py  --dataset kitti \
                --datapath $DATAPATH \
                --testlist ./filenames/kitti12_val.txt \
                --test_batch_size 1 \
                --model gwcnet-ca \
                --loadckpt your \


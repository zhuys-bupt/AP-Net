#!/usr/bin/env bash
set -x
DATAPATH="your"
python main.py  --dataset sceneflow \
                --datapath $DATAPATH \
                --trainlist ./filenames/sceneflow_train.txt \
                --testlist ./filenames/sceneflow_test.txt \
                --batch_size 12 \
                --test_batch_size 8 \
                --train_num_workers 12 \
                --test_num_workers 4 \
                --epochs 64 \
                --lrepochs "20,32,40,48,56:2" \
                --model gwcnet-ca \
                --logdir youur \



















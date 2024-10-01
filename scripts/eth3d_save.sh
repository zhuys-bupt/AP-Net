#!/usr/bin/env bash
set -x
DATAPATH="your"
python save_disp.py --dataset eth3d \
                    --datapath $DATAPATH \
                    --testlist ./filenames/eth3d_test.txt \
                    --model gwcnet-ca \
                    --loadckpt your \
                    --savedir your \

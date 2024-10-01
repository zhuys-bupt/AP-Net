#!/usr/bin/env bash
set -x
DATAPATH="your"
python save_disp.py --datapath $DATAPATH \
                    --testlist ./filenames/kitti12_test.txt \
                    --model gwcnet-ca \
                    --loadckpt your \
                    --savedir your \




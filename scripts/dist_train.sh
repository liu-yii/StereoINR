#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-5053}

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}

# torch 2.0
#PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
#CUDA_VISIBLE_DEVICES=2,3 \
#python -m torch.distributed.launch --use_env --nproc_per_node=$GPUS --master_port=$PORT \
#    basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}
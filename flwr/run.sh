#!/bin/bash


python server.py --world_size 4 --dataset_id 1 --lr 0.001 &

python client.py --dataset_id 1 --epochs 30 --batch_size 32 --lr 0.001 &
python client.py --dataset_id 2 --epochs 30 --batch_size 32 --lr 0.001 &
python client.py --dataset_id 3 --epochs 30 --batch_size 32 --lr 0.001 &
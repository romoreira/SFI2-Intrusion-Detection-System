#!/bin/bash


python server.py --world_size 4 --dataset_id 2 --lr 0.001 &

python client.py --world_size 4 --rank 1 --epoch 50 --dataset_id 1 --batch_size 32 --lr 0.001 &
python client.py --world_size 4 --rank 2 --epoch 50 --dataset_id 2 --batch_size 32 --lr 0.001 &
python client.py --world_size 4 --rank 3 --epoch 50 --dataset_id 3 --batch_size 32 --lr 0.001 &
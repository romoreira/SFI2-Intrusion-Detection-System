#!/bin/bash


python server.py --world_size 4 --dataset_id 1 --lr 0.001 &

python client.py --world_size 4 --rank 1 --epoch 100 --dataset_id 1 --batch_size 32 --lr 0.001 &
python client.py --world_size 4 --rank 2 --epoch 200 --dataset_id 2 --batch_size 32 --lr 0.0001 &
python client.py --world_size 4 --rank 3 --epoch 300 --dataset_id 3 --batch_size 32 --lr 0.001 &
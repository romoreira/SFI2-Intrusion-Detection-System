#!/bin/bash


python server.py --world_size 3 --dataset_id 2&

python client.py --world_size 3 --rank 1 --epoch 100 --dataset_id 1 --batch_size 32 --lr 0.001 &
python client.py --world_size 3 --rank 2 --epoch 100 --dataset_id 1 --batch_size 32 --lr 0.001 &
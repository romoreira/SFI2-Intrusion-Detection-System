#!/bin/bash


python3 server.py --world_size 4 --dataset_id 1 --lr 0.0001 &
sleep 3

python3 client.py --dataset_id 1 --epochs 50 --batch_size 32 --lr 0.0007415351948165897 --optim RMSprop &
python3 client.py --dataset_id 2 --epochs 50 --batch_size 32 --lr 0.002368577643931122 --optim Adam &
python3 client.py --dataset_id 3 --epochs 50 --batch_size 32 --lr 0.0024495168348176374 --optim Adam &


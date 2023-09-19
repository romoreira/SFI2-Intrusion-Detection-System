#!/bin/bash


python3 server.py --world_size 4 --dataset_id 1 --lr 0.0001 &
sleep 3

python3 client.py --dataset_id 1 --epochs 50 --batch_size 32 --lr 0.0007415351948165897 --optim RMSprop &
python3 client.py --dataset_id 2 --epochs 50 --batch_size 32 --lr 0.00029258198139112075 --optim RMSprop &
python3 client.py --dataset_id 3 --epochs 50 --batch_size 32 --lr 0.0026589777488326975 --optim Adam &


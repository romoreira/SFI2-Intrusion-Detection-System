#!/bin/bash


python3 unb_server.py --world_size 4 --dataset_id 1 --lr 0.001 &
sleep 3

#python3 unb_client.py --dataset_id 1 --epochs 20 --batch_size 32 --lr 0.01 --optim SGD &
python3 unb_client.py --dataset_id 2 --epochs 5 --batch_size 32 --lr 0.001 --optim SGD &
python3 unb_client.py --dataset_id 3 --epochs 5 --batch_size 32 --lr 0.001 --optim SGD &
python3 unb_client.py --dataset_id 4 --epochs 5 --batch_size 32 --lr 0.001 --optim SGD &
#python3 unb_client.py --dataset_id 5 --epochs 20 --batch_size 32 --lr 0.01 --optim SGD &
#python3 unb_client.py --dataset_id 6 --epochs 20 --batch_size 32 --lr 0.01 --optim SGD &
#python3 unb_client.py --dataset_id 7 --epochs 20 --batch_size 32 --lr 0.01 --optim SGD &


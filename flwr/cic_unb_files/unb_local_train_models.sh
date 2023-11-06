#!/bin/bash

# Crie o diretório de logs se ele não existir
mkdir -p ../results/cic-unb/logs/local_training

log_file="../results/cic-unb/logs/local_training/log_$i.txt"  # Nome do arquivo de log com caminho

  # Iniciar os clientes e redirecionar a saída para o mesmo arquivo de log
python3 unb_nn_local.py --dataset_id 1 --epochs 10 --batch_size 32 --lr 0.0003074258400864182 --optim Adam >> "$log_file" &
python3 unb_nn_local.py --dataset_id 2 --epochs 10 --batch_size 32 --lr 0.0005025961155459187 --optim RMSprop >> "$log_file" &
python3 unb_nn_local.py --dataset_id 3 --epochs 10 --batch_size 32 --lr 0.00010603472201401003 --optim RMSprop >> "$log_file" &
python3 unb_nn_local.py --dataset_id 4 --epochs 10 --batch_size 32 --lr 0.00013936442920558617 --optim Adam >> "$log_file" &
python3 unb_nn_local.py --dataset_id 5 --epochs 10 --batch_size 32 --lr 0.000587441102433824 --optim RMSprop >> "$log_file" &
python3 unb_nn_local.py --dataset_id 6 --epochs 10 --batch_size 32 --lr 0.0006052967400865347 --optim SGD >> "$log_file" &
python3 unb_nn_local.py --dataset_id 7 --epochs 10 --batch_size 32 --lr 0.00012091571705782663 --optim Adam >> "$log_file" &
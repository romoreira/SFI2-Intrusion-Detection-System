#!/bin/bash

# Crie o diretório de logs se ele não existir
mkdir -p ../results/cic-unb/logs

# Loop para executar o script 10 vezes
for i in {1..10}; do
  log_file="../results/cic-unb/logs/log_$i.txt"  # Nome do arquivo de log com caminho

  # Iniciar o servidor e redirecionar a saída para o arquivo de log
  python3 unb_server.py --world_size 8 --dataset_id 1 --lr 0.001 &> "$log_file" &
  server_pid=$!  # Salvar o PID do servidor

  sleep 10  # Aguardar um pouco antes de iniciar os clientes

  # Iniciar os clientes e redirecionar a saída para o mesmo arquivo de log
  python3 unb_client.py --dataset_id 1 --epochs 10 --batch_size 32 --lr 0.0003074258400864182 --optim Adam >> "$log_file" &
  python3 unb_client.py --dataset_id 2 --epochs 10 --batch_size 32 --lr 0.0005025961155459187 --optim RMSprop >> "$log_file" &
  python3 unb_client.py --dataset_id 3 --epochs 10 --batch_size 32 --lr 0.00010603472201401003 --optim RMSprop >> "$log_file" &
  python3 unb_client.py --dataset_id 4 --epochs 10 --batch_size 32 --lr 0.00013936442920558617 --optim Adam >> "$log_file" &
  python3 unb_client.py --dataset_id 5 --epochs 10 --batch_size 32 --lr 0.000587441102433824 --optim RMSprop >> "$log_file" &
  python3 unb_client.py --dataset_id 6 --epochs 10 --batch_size 32 --lr 0.0006052967400865347 --optim SGD >> "$log_file" &
  python3 unb_client.py --dataset_id 7 --epochs 10 --batch_size 32 --lr 0.00012091571705782663 --optim Adam >> "$log_file" &

  # Esperar que o servidor e os clientes terminem antes de prosseguir
  wait "$server_pid"
  ./clear_unb.sh
  echo "FIM Ciclo"
  sleep 60
done
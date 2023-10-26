#!/bin/bash

# Crie o diretório de logs se ele não existir
mkdir -p ./results/cic-unb/logs

# Loop para executar o script 10 vezes
for i in {2..10}; do
  log_file="./results/cic-unb/logs/log_$i.txt"  # Nome do arquivo de log com caminho

  # Iniciar o servidor e redirecionar a saída para o arquivo de log
  python3 unb_server.py --world_size 5 --dataset_id 1 --lr 0.001 &> "$log_file" &
  server_pid=$!  # Salvar o PID do servidor

  sleep 10  # Aguardar um pouco antes de iniciar os clientes

  # Iniciar os clientes e redirecionar a saída para o mesmo arquivo de log
  python3 unb_client.py --dataset_id 1 --epochs 10 --batch_size 32 --lr 0.001 --optim SGD >> "$log_file" &
  python3 unb_client.py --dataset_id 2 --epochs 20 --batch_size 32 --lr 0.0002151122320208885 --optim Adam >> "$log_file" &
  python3 unb_client.py --dataset_id 3 --epochs 10 --batch_size 32 --lr 0.001 --optim SGD >> "$log_file" &
  python3 unb_client.py --dataset_id 5 --epochs 10 --batch_size 32 --lr 0.001 --optim SGD >> "$log_file" &

  # Esperar que o servidor e os clientes terminem antes de prosseguir
  wait "$server_pid"
  ./clear_unb.sh
  echo "FIM Primeiro Ciclo"
  sleep 60
done
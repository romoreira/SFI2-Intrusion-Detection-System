#!/bin/bash

# Mate todos os processos chamados "client.py" em segundo plano
pkill -f "server.py"
while pkill -f "client.py"; do
  sleep 1  # Aguarde um segundo antes de verificar novamente
done

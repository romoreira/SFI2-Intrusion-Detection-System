#!/bin/bash

# Mata todos os processos relacionados ao server.py
pkill -f "python3 server.py"

# Mata todos os processos relacionados ao client.py
pkill -f "python3 client.py"

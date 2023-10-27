#!/bin/bash

for i in 1 2 3 5; do
  python3 unb_nn_local.py --dataset_id $i
  echo "Execução $i: python3 local.py --dataset_id $i"
done
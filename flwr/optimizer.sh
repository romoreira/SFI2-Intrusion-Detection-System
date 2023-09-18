#!/bin/bash

for ((i=1; i<=3; i++))
do
  dataset_id=$((i))
  python3 local.py --dataset_id $dataset_id
  echo "Execução $i: python3 local.py --dataset_id $dataset_id"
done
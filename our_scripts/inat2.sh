#!/bin/bash

# for num in 0. 0.06 0.07 0.08 0.09 0.1
for num in 0. 0.01 0.02
do 
    python -u main.py -d inat2018 -m in21k_vit_b16_peft loss_type LA reg True temper 15. gpu 1 weight $num &> inat_temp15/inat_$num.txt
done
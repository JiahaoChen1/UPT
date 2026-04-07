#!/bin/bash

# for num in 0. 0.06 0.07 0.08 0.09 0.1
for num in  0.03 0.04
do 
    python -u main.py -d inat2018 -m in21k_vit_b16_peft loss_type LA reg True temper 10. weight $num &> inat_temp10/inat_$num.txt
done
#!/bin/bash

# for num in 0.06 0.07 0.08 0.09 0.1
for num in 0.01 0.03 0.06
do 
    python main.py -d imagenet_lt -m in21k_vit_b16_peft loss_type CE reg True temper 15. gpu 1 weight $num &> 0inter/imagenet_temp15_base/imagenet_$num.txt
done
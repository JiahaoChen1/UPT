#!/bin/bash

# for num in 0.01 0.02 0.03 0.04
# do 
#     for t in 1.1 1.2 1.3 1.4 1.5
#     do
#         python -u main.py -d cifar100_ir100 -m in21k_vit_b16_peft loss_type LA reg True gpu 0 weight $num temper $t &> 00_130_all_res/la_cifar/cifar100_num{$num}_{$t}.txt
#     done
# done
for num in 0.16 0.18
do
    python -u main.py -d cifar100_ir100 -m in21k_vit_b16_peft loss_type LA reg True gpu 0 weight $num temper 1.3  &> 00_130_all_res/la_cifar/cifar100_num{$num}_1.0.txt
done


# for num in 0.005 0.006 0.007 0.008 0.009
# do
#     python -u main.py -d cifar100_ir100 -m clip_vit_b16_peft loss_type LA reg True gpu 0 weight $num temper 1.1 &> 00_130_all_res/la_cifar/cifar100_num{$num}_1.1.txt
# done

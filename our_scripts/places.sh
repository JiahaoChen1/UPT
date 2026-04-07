#!/bin/bash


# for num in 0.01 0.02 0.03 0.04
# do 
#     for t in 1.1 1.2 1.3 1.4 1.5
#     do
#         python -u main.py -d places_lt -m in21k_vit_b16_peft loss_type LA reg True gpu 2 weight $num temper $t &> 00_130_all_res/la_places/places_num{$num}_{$t}.txt
#     done
# done

for num in 0. 0.01 0.02 0.03 0.04
do
    python -u main.py -d places_lt -m clip_vit_b16_peft loss_type LA reg True gpu 0 weight $num temper 1.3 &> 00_130_all_res/la_places/places_num{$num}_{$t}.txt
done

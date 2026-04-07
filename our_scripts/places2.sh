#!/bin/bash

for num in 0.02 0.03
do 
    python main.py -d places_lt -m clip_vit_b16_peft loss_type LA reg True gpu 2 temper 5. weight $num num_epochs 10 &> 0lift_res/places_temp5/jitter_places_$num.txt
done
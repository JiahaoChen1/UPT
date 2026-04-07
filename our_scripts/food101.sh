

# for num in 0. 0.04 0.05 0.06 0.07 0.08
# do 
#     python -u main.py -d food101_lt -m in21k_vit_b16_peft loss_type LDAM reg True weight $num gpu 0 temper 15. &> 00_all_res/base_food/temp15/food_$num.txt
# done

# for num in 0.04 0.05 0.06 0.07 0.08
# do 
#     python -u main.py -d food101_lt -m in21k_vit_b16_peft loss_type LDAM reg True weight $num gpu 0 temper 20. &> 00_all_res/base_food/temp20/food_$num.txt
# done


for num in 0. 0.04 0.05 0.06 
do 
    python -u main.py -d food101_lt -m in21k_vit_b16_peft loss_type LA reg True weight $num gpu 0 temper 10. num_epochs 20 &> 00_all_res/la_food/temp10/food_$num.txt
done

for num in 0.04 0.05 0.06 
do 
    python -u main.py -d food101_lt -m in21k_vit_b16_peft loss_type LA reg True weight $num gpu 0 temper 15. num_epochs 20 &> 00_all_res/la_food/temp15/food_$num.txt
done

for num in 0. 0.04 0.05 0.06 
do 
    python -u main.py -d food101_lt -m in21k_vit_b16_peft loss_type SADE classifier ExpertsClassifier reg True weight $num gpu 0 temper 10. num_epochs 20 &> 00_all_res/sade_food/temp10/food_$num.txt
done


for num in 0.04 0.05 0.06 
do 
    python -u main.py -d food101_lt -m in21k_vit_b16_peft loss_type SADE classifier ExpertsClassifier reg True weight $num gpu 0 temper 15. num_epochs 20 &> 00_all_res/sade_food/temp15/food_$num.txt
done







for num in 0.04 0.05 0.06 
do 
    python -u main.py -d fgcv_lt -m in21k_vit_b16_peft loss_type LA reg True weight $num gpu 0 temper 10. batch_size 32 micro_batch_size 32 num_epochs 40 lr 0.1  &> 00_all_res/la_fgcv/temp10/fgcv_$num.txt
done

for num in 0.04 0.05 0.06 
do 
    python -u main.py -d fgcv_lt -m in21k_vit_b16_peft loss_type LA reg True weight $num gpu 0 temper 15. batch_size 32 micro_batch_size 32 num_epochs 40 lr 0.1 &> 00_all_res/la_fgcv/temp15/fgcv_$num.txt
done

for num in 0. 0.04 0.05 0.06 
do 
    python -u main.py -d fgcv_lt -m in21k_vit_b16_peft loss_type SADE classifier ExpertsClassifier reg True weight $num gpu 0 temper 10. batch_size 32 micro_batch_size 32 num_epochs 40 lr 0.1  &> 00_all_res/sade_fgcv/temp10/fgcv_$num.txt
done


for num in 0.04 0.05 0.06 
do 
    python -u main.py -d fgcv_lt -m in21k_vit_b16_peft loss_type SADE classifier ExpertsClassifier reg True weight $num gpu 0 temper 15. batch_size 32 micro_batch_size 32 num_epochs 40 lr 0.1 &> 00_all_res/sade_fgcv/temp15/fgcv_$num.txt
done



for num in 0. 0.04 0.05 0.06 0.07 0.08
do 
    python -u main.py -d deepfish -m in21k_vit_b16_peft loss_type LA reg True weight $num gpu 0 temper 10. num_epochs 5 &> 00_all_res/la_fish/temp10/fish_$num.txt
done

for num in  0.04 0.05 0.06 0.07 0.08
do 
    python -u main.py -d deepfish -m in21k_vit_b16_peft loss_type LA reg True weight $num gpu 0 temper 15. num_epochs 5 &> 00_all_res/la_fish/temp15/fish_$num.txt
done

for num in 0. 0.04 0.05 0.06 0.07 0.08
do 
    python -u main.py -d deepfish -m in21k_vit_b16_peft loss_type SADE classifier ExpertsClassifier reg True weight $num gpu 0 temper 10. num_epochs 5 &> 00_all_res/sade_fish/temp10/fish_$num.txt
done


for num in  0.04 0.05 0.06 0.07 0.08
do 
    python -u main.py -d deepfish -m in21k_vit_b16_peft loss_type SADE classifier ExpertsClassifier reg True weight $num gpu 0 temper 15. num_epochs 5 &> 00_all_res/sade_fish/temp15/fish_$num.txt
done
#!/bin/bash

# for num in 0.0
# do 
#     python -u main.py -d cifar100_ir100 -m in21k_vit_b16_peft loss_type CE reg True weight $num gpu 1 temper 5. vpt_deep False &> 0inter2/cifar100_temp5/linear_cifar100_$num.txt
# done



python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name brightness.npy  &> 0robust/our/brightness.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name defocus_blur.npy  &> 0robust/our/defocus_blur.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name fog.npy  &> 0robust/our/fog.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name gaussian_blur.npy  &> 0robust/our/gaussian_blur.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name glass_blur.npy  &> 0robust/our/glass_blur.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name jpeg_compression.npy  &> 0robust/our/jpeg_compression.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name motion_blur.npy  &> 0robust/our/motion_blur.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name shot_noise.npy  &> 0robust/our/shot_noise.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name spatter.npy  &> 0robust/our/spatter.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name zoom_blur.npy  &> 0robust/our/zoom_blur.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name contrast.npy  &> 0robust/our/contrast.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name elastic_transform.npy  &> 0robust/our/elastic_transform.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name frost.npy  &> 0robust/our/frost.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name gaussian_noise.npy  &> 0robust/our/gaussian_noise.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name impulse_noise.npy  &> 0robust/our/impulse_noise.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name pixelate.npy   &> 0robust/our/pixelate.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name saturate.npy  &> 0robust/our/saturate.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name snow.npy  &> 0robust/our/snow.txt
python main.py -d cifar100_ir100 -m in21k_vit_b16_peft test_only True  gpu 1 model_dir /data00/jiahao/PEL-main/output/cifar100_ir100_in21k_vit_b16_peft_loss_type_LA_reg_True_weight_0.06_gpu_0_temper_15. sub_name speckle_noise.npy  &> 0robust/our/speckle_noise.txt
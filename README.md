# Uniformity Preserving Transfer (UPT)

This repository contains the code and paper source for:

**Uniformity Preserving Transfer for Visual Prompt Tuning under Long-tailed Distribution**

The core goal is to improve tail-class performance in long-tailed recognition by preserving and transferring feature-space uniformity from a frozen foundation model to a prompt-tuned model.

## Project Structure

- `paper.tex`: main LaTeX source of the paper
- `main.py`: training/testing entry point
- `trainer.py`: training logic and evaluation pipeline
- `configs/data/`: dataset configs (`*.yaml`)
- `configs/model/`: model/backbone + PEFT configs (`*.yaml`)
- `datasets/`: dataset loaders and split files
- `models/`: model definitions and PEFT modules
- `utils/`: config, logging, losses, evaluator, samplers, etc.
- `our_scripts/` and `scripts/`: experiment shell scripts

## Environment Setup

It is recommended to use Python 3.8+ with CUDA-enabled PyTorch.

Example setup:

```bash
conda create -n upt python=3.9 -y
conda activate upt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install yacs numpy
```

If your environment already has PyTorch, only install missing packages.


## Training

Main command format:

```bash
python main.py -d <data_config_name> -m <model_config_name> [key value ...]
```

Example (similar to provided scripts):

```bash
python -u main.py -d cifar100_ir100 -m in21k_vit_b16_peft \
  loss_type LA reg True gpu 0 weight 0.16 temper 1.3
```

Another example:

```bash
python -u main.py -d imagenet_lt -m in21k_vit_b16_peft \
  loss_type LA reg True gpu 1 weight 0.02 temper 1.0
```


## Evaluation

Run test-only mode by loading a trained checkpoint directory:

```bash
python main.py -d <data_config_name> -m <model_config_name> \
  test_only True model_dir <output_subdir>
```

Related flags in config include:

- `test_only`
- `test_train`
- `knn_only`
- `zero_shot`

## Output

By default, outputs are saved to:

```text
./output/<data>_<model>_<opts...>
```

Logs and checkpoints are written there.


## Notes

- This repository currently does not include a pinned `requirements.txt`.
- For reproducibility, keep a record of your CUDA, PyTorch, and driver versions.
- Some experiments may require large GPU memory and long training time on long-tailed datasets.

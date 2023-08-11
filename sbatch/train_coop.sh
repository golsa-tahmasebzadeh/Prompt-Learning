#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1


#eval "$(/nfs/home/tahmasebzadehg/miniconda3/bin/conda shell.bash hook)" # init conda
#conda activate py312
export MASTER_PORT=12813

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn /nfs/home/tahmasebzadehg/miniconda3/envs/py310/bin/python -u /nfs/home/tahmasebzadehg/prompt_learning/src/training/main_coop.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --val-frequency 1 \
    --batch-size=32 \
    --workers=8 \
    --precision amp \
    --seed 0 \
    --epochs 201 \
    --data-dir "/nfs/home/tahmasebzadehg/prompt_learning/data"\
    --pretrained openai \
    --lr 2e-03 \
    --wd 0.001 \
    --dataset-name "test" \
    --kg-init 'random'\
    --gt-label-name "gt_label"\
    --CLASS-TOKEN-POSITION 'front'\
    --CSC 'True'\
    --model ViT-B-32 \
    --N-CTX 16\
    --data_set_number '1'\
    --pre-fname '2023_08_02_TEST'

#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH -w devbox4
#SBATCH --job-name=~

#conda activate venv
export MASTER_PORT=12813

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn /nfs/home/tahmasebzadehg/prompt_learning/venv/bin/python -u /nfs/home/tahmasebzadehg/prompt_learning/src/training/main_coop.py \
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
    --dataset-name "EVENT" \
    --kg-init 'wikipedia'\
    --gt-label-name "gt_label"\
    --CLASS-TOKEN-POSITION 'front'\
    --CSC 'True'\
    --model ViT-B-32 \
    --N-CTX 16\
    --pre-fname '2023_08_07_EVENT'

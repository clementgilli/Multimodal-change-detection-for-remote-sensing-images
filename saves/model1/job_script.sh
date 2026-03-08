#!/bin/bash

#SBATCH --job-name=my_job             # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=P100
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=32G                    # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

MODEL="gradualexpansionunet"
ACTIVATION="silu"
INTERPOLATION_MODE="Bilinear"
LEARNING_MODE="residual"
LAMBDA_SAM=0.0
LR=1e-4
EPOCHS=10
BATCH_SIZE=8
NUM_WORKERS=6

PYTHON_EXEC="/home/infres/gilli-23/miniconda3/envs/remote_sensing/bin/python"

$PYTHON_EXEC src/train.py --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
    --model $MODEL --activation $ACTIVATION --interpolation_mode $INTERPOLATION_MODE \
    --learning_mode $LEARNING_MODE --lambda_sam $LAMBDA_SAM --lr $LR

echo "Job finished at: $(date)"

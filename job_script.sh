echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

MODEL="gradualexpansionunett"
ACTIVATION="silu"
INTERPOLATION_MODE="Bilinear"
LEARNING_MODE="residual"
LAMBDA_SAM=1.0
LR=1e-3
EPOCHS=100
BATCH_SIZE=8
NUM_WORKERS=4

conda activate remote_sensing

srun python src/train.py --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
    --model $MODEL --activation $ACTIVATION --interpolation_mode $INTERPOLATION_MODE \
    --learning_mode $LEARNING_MODE --lambda_sam $LAMBDA_SAM --lr $LR

echo "Job finished at: $(date)"
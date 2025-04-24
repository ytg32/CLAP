#!/bin/bash
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --job-name=my_gpu_job
#SBATCH --output=my_gpu_job_%j.out
#SBATCH --error=my_gpu_job_%j.err

# Load necessary modules
module purge
module load CUDA/12.4.0
module load Python/3.9.6-GCCcore-11.2.0

# Activate Python virtual environment
source /cluster/work/boraa/beans/env/bin/activate

# Navigate to project directory
cd /cluster/work/boraa/CLAP/src/laion_clap

# Run the training script
python -m training.main \
    --save-frequency 5 \
    --save-top-performance 2 \
    --save-most-recent \
    --amodel BEATs \
    --tmodel modern_bert \
    --dataset-type="webdataset" \
    --datasetnames "audiocaps" "iNatural" "watkins" \
    --datasetinfos "train" "val" "test"\
    --datasetpath /cluster/work/boraa/data/AnimalSpeak/webdataset \
    --precision="fp32" \
    --warmup 3200 \
    --batch-size=128 \
    --lr=5e-5 \
    --wd=0.0 \
    --epochs=45 \
    --report-to "wandb" \
    --wandb-notes "10.16-clap-dataset-1#-beats-modern_bert-lr 5e-5 v3" \
    --top-k-checkpoint-select-metric="mAP@1" \
    --top-k-checkpoint-select-dataset="iNatural-val" \
    --seed 3407 \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --audio-ext "wav" \
    --audio-max-len 128_000
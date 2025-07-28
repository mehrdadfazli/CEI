#!/bin/bash
#SBATCH --job-name=CEI_CHAIR_DYN
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=0-2:00:00
#SBATCH --output=/projects/zzhu20/Mehrdad/CEI/results/CHAIR/prelim_dynamic_ctx_embed/job_%j/slurm.out
#SBATCH --error=/projects/zzhu20/Mehrdad/CEI/results/CHAIR/prelim_dynamic_ctx_embed/job_%j/slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfazli@gmu.edu

# Initialize Conda
source /projects/zzhu20/Mehrdad/miniforge/bin/activate
conda activate lvlm || { echo "Failed to activate lvlm environment"; exit 1; }

# Set experiment directory
#EXP_DIR="/projects/zzhu20/Mehrdad/CEI/results/CHAIR/prelim_dynamic_ctx_embed/job_${SLURM_JOB_ID}"
# mkdir -p $EXP_DIR

# Copy the configuration file to the experiment directory
# cp $CONFIG_FILE $EXP_DIR/config.json

# Run the Bash script with the copied config file and exp_dir
bash /projects/zzhu20/Mehrdad/CEI/run_chair_dyn.sh $CONFIG_FILE

#- Run this slurm file using below:
#- sbatch --export=CONFIG_FILE=/projects/zzhu20/Mehrdad/CEI/configs/config1.json chair_dyn.slurm
#!/bin/bash 
#SBATCH --job-name=djepa          # Name of your job 
#SBATCH --output=/zfsauton2/home/yiqiw2/slurmlogs/job_%j.out    # Standard output log (%j = Job ID) 
#SBATCH --error=/zfsauton2/home/yiqiw2/slurmlogs/job_%j.err     # Error log 
#SBATCH --partition=legacy             # (General, Debug, Preempt or Cpu, legacy)
#SBATCH --qos=qos_legacy               # Matches the partition for guaranteed priority
#SBATCH --ntasks=1                      # Number of tasks 
#SBATCH --cpus-per-task=8               # CPU cores per task 
#SBATCH --mem=64G                       # Memory (RAM) limit 
#SBATCH --time=2-00:00:00                       # Time limit (D-HH:MM:SS) 
#SBATCH --gres=gpu:rtx_2080_ti:1              # Request 1 rtx_2080_ti, v100, a6000 GPU 

#Envs  
source $(conda info --base)/etc/profile.d/conda.sh
conda activate djepa

srun python djepa.py
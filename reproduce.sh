#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=00:50:00
#SBATCH --export=NONE
#SBATCH --job-name=reproduce_check

unset SLURM_EXPORT_ENV

if [ -z "$1" ]; then
    echo "ERROR: no run_id given. Submit like this: sbatch reproduce.sh <run_id>"
    exit 1
fi

config_path="experiments/configs/config_${1}.yaml"

if [ ! -f "$config_path" ]; then
    echo "ERROR: no config file found at ${config_path}"
    echo "Check that you typed the run_id correctly."
    exit 1
fi

module load python
source venv/bin/activate

python main.py --config "$config_path"
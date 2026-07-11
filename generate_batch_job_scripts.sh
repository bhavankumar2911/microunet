#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash generate_batch_jobs.sh <output_directory> <job_name_prefix> [config_folder]"
    echo "Example: bash generate_batch_jobs.sh batch-job-cyclic-lr cyclic_lr configs/cyclic_lr_baseline"
    exit 1
fi

output_directory="$1"
job_name_prefix="$2"
config_folder="${3:-configs/data_augmentation}"

mkdir -p "$output_directory"

existing_files=$(ls "${output_directory}"/*.batch-job.sh 2>/dev/null | wc -l)
if [ "$existing_files" -gt 0 ]; then
    echo "WARNING: ${existing_files} existing .batch-job.sh file(s) found in ${output_directory}. They will be overwritten."
fi

generate_batch_job() {
    local dataset_name="$1"
    local job_suffix="$2"
    local time_allocation="$3"
    local config_path="${config_folder}/${dataset_name}.yaml"
    local job_name="${job_name_prefix}_${dataset_name}_${job_suffix}"

    cat > "${output_directory}/${dataset_name}.batch-job.sh" << EOF
#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=${time_allocation}
#SBATCH --export=NONE
#SBATCH --job-name=${job_name}

unset SLURM_EXPORT_ENV

module load python
source venv/bin/activate

python main.py --config ${config_path}
EOF
}

#                  dataset_name      job_suffix   time
generate_batch_job "acdc"           "batch-job"  "00:50:00"
generate_batch_job "btcv"           "batch-job"  "02:58:00"
generate_batch_job "cellnuclei"     "batch-job"  "00:20:00"
generate_batch_job "chaos"          "batch-job"  "00:57:00"
generate_batch_job "emsegmentation" "batch-job"  "00:50:00"
generate_batch_job "fhpsaop"        "batch-job"  "01:15:00"
generate_batch_job "isic2016"       "batch-job"  "00:26:00"
generate_batch_job "mmwhsmr"        "batch-job"  "01:35:00"
generate_batch_job "nuset"          "batch-job"  "00:39:00"
generate_batch_job "usforkidney"    "batch-job"  "00:51:00"
generate_batch_job "wbc"            "batch-job"  "00:32:00"
generate_batch_job "yeaz"           "batch-job"  "00:24:00"

echo "Generated ${output_directory}:"
ls -1 "${output_directory}"
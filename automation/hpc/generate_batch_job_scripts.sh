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

# ==============================================================================
# EXISTING BASELINE DATASETS
# ==============================================================================
#                  dataset_name      job_suffix   time
generate_batch_job "acdc"           "batch-job"  "00:50:00"
generate_batch_job "btcv"           "batch-job"  "02:50:00"
generate_batch_job "cellnuclei"     "batch-job"  "00:40:00"
generate_batch_job "chaos"          "batch-job"  "00:57:00"
generate_batch_job "emsegmentation" "batch-job"  "00:50:00"
generate_batch_job "fhpsaop"        "batch-job"  "01:20:00"
generate_batch_job "isic2016"       "batch-job"  "00:40:00"
generate_batch_job "mmwhsmr"        "batch-job"  "01:30:00"
generate_batch_job "nuset"          "batch-job"  "01:10:00"
generate_batch_job "usforkidney"    "batch-job"  "01:50:00"
generate_batch_job "wbc"            "batch-job"  "00:15:00"
generate_batch_job "yeaz"           "batch-job"  "00:20:00"

# ==============================================================================
# NEW EXTENDED DATASETS (Walltimes extrapolated from Train Sample sizes)
# ==============================================================================
# Micro-scale datasets (N <= 100 train samples) -> 15 min baseline floor
generate_batch_job "chasedb1"       "batch-job"  "00:40:00"   # N=19
generate_batch_job "chuac"          "batch-job"  "00:15:00"   # N=21
generate_batch_job "deepbacs"       "batch-job"  "00:30:00"   # N=17
generate_batch_job "drive"          "batch-job"  "00:15:00"   # N=18
generate_batch_job "idrib"          "batch-job"  "00:40:00"   # N=47
generate_batch_job "pandental"      "batch-job"  "00:15:00"   # N=81
generate_batch_job "nuclei"         "batch-job"  "00:15:00"   # N=98
generate_batch_job "dca1"           "batch-job"  "00:25:00"   # N=93
generate_batch_job "tnbcnuclei"     "batch-job"  "00:30:00"   # N=35

# Small-scale datasets (100 < N <= 500 train samples) -> 20-25 mins
generate_batch_job "bbbc010"        "batch-job"  "00:15:00"   # N=70 (Micro-border)
generate_batch_job "uwskincancer"   "batch-job"  "00:15:00"   # N=143
generate_batch_job "monusac"        "batch-job"  "00:15:00"   # N=188
generate_batch_job "m2caiseg"       "batch-job"  "00:20:00"   # N=245
generate_batch_job "robotool"       "batch-job"  "00:20:00"   # N=350
generate_batch_job "busi"           "batch-job"  "00:25:00"   # N=452

# Medium-scale datasets (500 < N <= 1000 train samples) -> 30-40 mins
generate_batch_job "abdomenus"      "batch-job"  "00:50:00"   # N=569
generate_batch_job "bkaiigh"        "batch-job"  "00:35:00"   # N=700
generate_batch_job "kvasir"         "batch-job"  "00:35:00"   # N=700
generate_batch_job "cystofluid"     "batch-job"  "00:35:00"   # N=703
generate_batch_job "polypgen"       "batch-job"  "00:40:00"   # N=984
generate_batch_job "brifiseg"       "batch-job"  "00:40:00"   # N=1005
generate_batch_job "polyp"          "batch-job"  "00:40:00"   # Local standard frames

# Large-scale datasets (1000 < N <= 3000 train samples) -> 45 min to 1 hr 20 min
generate_batch_job "promise12"      "batch-job"  "00:45:00"   # N=1031
generate_batch_job "ultrasoundnerve" "batch-job" "00:55:00"   # N=1651
generate_batch_job "covidquex"      "batch-job"  "01:10:00"   # N=1864
generate_batch_job "mosmedplus"     "batch-job"  "01:00:00"   # N=1910
generate_batch_job "isic2018"       "batch-job"  "01:15:00"   # N=2594
generate_batch_job "bagls"          "batch-job"  "02:00:00"   # Large local endoscopy series

# Massive-scale datasets (N > 3000 train samples) -> Scaled safely upward
generate_batch_job "dynamicnuclear" "batch-job"  "02:15:00"   # N=4950
generate_batch_job "covid19radio"   "batch-job"  "04:00:00"   # N=14814 (Massive X-Ray set)
#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: bash run.sh <batch_job_scripts_directory>"
    echo "Example: bash run.sh batch-job/naive"
    exit 1
fi

batch_job_scripts_directory="$(cd "$1" && pwd)"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root_directory="$(dirname "$script_directory")"
cd "$project_root_directory"

batch_job_scripts_in_order=(
    # ==========================================================================
    # 3D Patient-Grouped Volumetric Slices (IMed-361M)
    # ==========================================================================
    "acdc.batch-job.sh"
    "btcv.batch-job.sh"
    "chaos.batch-job.sh"
    "mmwhsmr.batch-job.sh"

    # ==========================================================================
    # 2D Image Benchmarks (Alphabetical Order)
    # ==========================================================================
    "abdomenus.batch-job.sh"
    "bbbc010.batch-job.sh"
    "bkaiigh.batch-job.sh"
    "brifiseg.batch-job.sh"
    "busi.batch-job.sh"
    "cellnuclei.batch-job.sh"
    "chasedb1.batch-job.sh"
    "chuac.batch-job.sh"
    "covid19radio.batch-job.sh"
    "covidquex.batch-job.sh"
    "cystofluid.batch-job.sh"
    "dca1.batch-job.sh"
    "deepbacs.batch-job.sh"
    "drive.batch-job.sh"
    "dynamicnuclear.batch-job.sh"
    "emsegmentation.batch-job.sh"
    "fhpsaop.batch-job.sh"
    "idrib.batch-job.sh"
    "isic2016.batch-job.sh"
    "isic2018.batch-job.sh"
    "kvasir.batch-job.sh"
    "m2caiseg.batch-job.sh"
    "monusac.batch-job.sh"
    "mosmedplus.batch-job.sh"
    "nuclei.batch-job.sh"
    "nuset.batch-job.sh"
    "pandental.batch-job.sh"
    "polypgen.batch-job.sh"
    "promise12.batch-job.sh"
    "robotool.batch-job.sh"
    "tnbcnuclei.batch-job.sh"
    "ultrasoundnerve.batch-job.sh"
    "usforkidney.batch-job.sh"
    "uwskincancer.batch-job.sh"
    "wbc.batch-job.sh"
    "yeaz.batch-job.sh"
)

for batch_job_script_name in "${batch_job_scripts_in_order[@]}"; do
    if [ -f "$batch_job_scripts_directory/$batch_job_script_name" ]; then
        echo "Submitting $batch_job_script_name"
        sbatch "$batch_job_scripts_directory/$batch_job_script_name"
    else
        echo "WARNING: Script not found: $batch_job_script_name (Skipping)"
    fi
done
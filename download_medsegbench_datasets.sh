#!/bin/bash

# Downloads all new MedSegBench datasets (28 datasets, 256px resolution).
# Already-existing files are skipped unless you pass --force as the first argument.
# Usage:
#   bash download_medsegbench_datasets.sh           # skip existing files
#   bash download_medsegbench_datasets.sh --force   # re-download everything

force_download=false
if [ "$1" = "--force" ]; then
    force_download=true
fi

download_dataset() {
    local folder_name="$1"
    local npz_filename="$2"
    local download_url="$3"
    local target_path="data/${folder_name}/${npz_filename}"

    mkdir -p "data/${folder_name}"

    if [ -f "$target_path" ] && [ "$force_download" = false ]; then
        echo "SKIP  ${target_path} (already exists)"
        return
    fi

    echo "GET   ${target_path}"
    wget -q --show-progress -O "$target_path" "$download_url"
    if [ $? -eq 0 ]; then
        echo "DONE  ${target_path}"
    else
        echo "FAIL  ${target_path}"
    fi
}

#              folder_name        npz_filename                      url
download_dataset "AbdomenUS"      "abdomenus_256.npz"               "https://zenodo.org/records/13358372/files/abdomenus_256.npz?download=1"
download_dataset "Bbbc010"        "bbbc010_256.npz"                 "https://zenodo.org/records/13358372/files/bbbc010_256.npz?download=1"
download_dataset "BkaiIgh"        "bkai-igh_256.npz"                "https://zenodo.org/records/13358372/files/bkai-igh_256.npz?download=1"
download_dataset "BriFiSeg"       "brifiseg_256.npz"                "https://zenodo.org/records/13358372/files/brifiseg_256.npz?download=1"
download_dataset "Busi"           "busi_256.npz"                    "https://zenodo.org/records/13358372/files/busi_256.npz?download=1"
download_dataset "ChaseDB1"       "chasedb1_256.npz"                "https://zenodo.org/records/13358372/files/chasedb1_256.npz?download=1"
download_dataset "Chuac"          "chuac_256.npz"                   "https://zenodo.org/records/13358372/files/chuac_256.npz?download=1"
download_dataset "Covid19Radio"   "covid19radio_256.npz"            "https://zenodo.org/records/13358372/files/covid19radio_256.npz?download=1"
download_dataset "CovidQUEx"      "covidquex_256.npz"               "https://zenodo.org/records/13358372/files/covidquex_256.npz?download=1"
download_dataset "CystoFluid"     "cystoidfluid_256.npz"            "https://zenodo.org/records/13358372/files/cystoidfluid_256.npz?download=1"
download_dataset "Dca1"           "dca1_256.npz"                    "https://zenodo.org/records/13358372/files/dca1_256.npz?download=1"
download_dataset "Deepbacs"       "deepbacs_256.npz"                "https://zenodo.org/records/13358372/files/deepbacs_256.npz?download=1"
download_dataset "Drive"          "drive_256.npz"                   "https://zenodo.org/records/13358372/files/drive_256.npz?download=1"
download_dataset "DynamicNuclear" "dynamicnuclear_256.npz"          "https://zenodo.org/records/13358372/files/dynamicnuclear_256.npz?download=1"
download_dataset "Idrib"          "idrib_256.npz"                   "https://zenodo.org/records/13358372/files/idrib_256.npz?download=1"
download_dataset "Isic2018"       "isic2018_256.npz"                "https://zenodo.org/records/13358372/files/isic2018_256.npz?download=1"
download_dataset "Kvasir"         "kvasir_256.npz"                  "https://zenodo.org/records/13358372/files/kvasir_256.npz?download=1"
download_dataset "M2caiSeg"       "m2caiseg_256.npz"                "https://zenodo.org/records/13358372/files/m2caiseg_256.npz?download=1"
download_dataset "Monusac"        "monusac_256.npz"                 "https://zenodo.org/records/13358372/files/monusac_256.npz?download=1"
download_dataset "MosMedPlus"     "mosmedplus_256.npz"              "https://zenodo.org/records/13358372/files/mosmedplus_256.npz?download=1"
download_dataset "Nuclei"         "nuclei_256.npz"                  "https://zenodo.org/records/13358372/files/nuclei_256.npz?download=1"
download_dataset "Pandental"      "pandental_256.npz"               "https://zenodo.org/records/13358372/files/pandental_256.npz?download=1"
download_dataset "PolypGen"       "polypgen_256.npz"                "https://zenodo.org/records/13358372/files/polypgen_256.npz?download=1"
download_dataset "Promise12"      "promise12_256.npz"               "https://zenodo.org/records/13358372/files/promise12_256.npz?download=1"
download_dataset "RoboTool"       "robotool_256.npz"                "https://zenodo.org/records/13358372/files/robotool_256.npz?download=1"
download_dataset "TnbcNuclei"     "tnbcnuclei_256.npz"              "https://zenodo.org/records/13358372/files/tnbcnuclei_256.npz?download=1"
download_dataset "UltrasoundNerve" "ultrasoundnerve_256.npz"        "https://zenodo.org/records/13358372/files/ultrasoundnerve_256.npz?download=1"
download_dataset "UwSkinCancer"   "uwaterlooskincancer_256.npz"     "https://zenodo.org/records/13358372/files/uwaterlooskincancer_256.npz?download=1"

echo ""
echo "All downloads complete."
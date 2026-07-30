#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: bash generate_configs.sh <output_directory>"
    echo "Example: bash generate_configs.sh configs/cyclic_lr_baseline"
    exit 1
fi

output_directory="$1"
mkdir -p "$output_directory"

generate_yaml() {
    local dataset_name="$1"
    local data_root="$2"
    local input_channels="$3"
    local use_color_input="$4"
    local epochs="$5"

    local lowercase_name
    lowercase_name=$(echo "$dataset_name" | tr '[:upper:]' '[:lower:]')

    cat > "${output_directory}/${lowercase_name}.yaml" << EOF
architecture:
  encoder_channels: [20, 40]
  bottleneck_channels: 80
  kernel_size: 3
  normalization: "instance_norm"
  group_norm_num_groups: 8
  activation: h_swish
  upsampling_mode: nearest_neighbor
  use_residual_connections: false
  use_attention_gates: true
  dropout_probability: 0.0
  input_channels: ${input_channels}
  use_depthwise_separable_convolutions: true
  use_single_convolution_per_block: false
  skip_connection_mode: concat
  weight_initialization: "kaiming_normal"

training:
  use_cyclic_learning_rate: false
  cyclic_learning_rate_minimum: 1e-5
  cyclic_learning_rate_maximum: 1e-4
  cyclic_learning_rate_half_cycle_epochs: 2
  data_root: ${data_root}
  learning_rate: 0.001
  weight_decay: 1e-4
  batch_size: 32
  epochs: ${epochs}
  dataset: ${dataset_name}
  image_size: 256
  use_color_input: ${use_color_input}
  early_stopping_patience: 10
  early_stopping_minimum_improvement_delta: 0.001
  use_augmentation: false
  augmentation_apply_horizontal_flip: true
  augmentation_apply_vertical_flip: true
  augmentation_rotation_max_angle_degrees: 15
  hypothesis: "Depth-wise separable layers (20-40-80) + attention gate does not lose much in Dice."
  notes: "Depth-wise separable layers (20-40-80) + attention gate"
EOF
}

#            dataset_name       data_root               input_channels  use_color_input  epochs
generate_yaml "AbdomenUS"       "data/AbdomenUS"        1               "false"          100
generate_yaml "Acdc"            "data/Acdc"             1               "false"          100
generate_yaml "BAGLS"           "data/BAGLS"            1               "false"          100
generate_yaml "Bbbc010"         "data/Bbbc010"          1               "false"          150
generate_yaml "BkaiIgh"         "data/BkaiIgh"          3               "true"           100
generate_yaml "BriFiSeg"        "data/BriFiSeg"         1               "false"          100
generate_yaml "Btcv"            "data/Btcv"             1               "false"          100
generate_yaml "Busi"            "data/Busi"             1               "false"          100
generate_yaml "CellNuclei"      "data/CellNuclei"       3               "true"           150
generate_yaml "Chaos"           "data/Chaos"            1               "false"          150
generate_yaml "ChaseDB1"        "data/ChaseDB1"         3               "true"           100
generate_yaml "Chuac"           "data/Chuac"            1               "false"          100
generate_yaml "Covid19Radio"    "data/Covid19Radio"     1               "false"          100
generate_yaml "CovidQUEx"       "data/CovidQUEx"        1               "false"          100
generate_yaml "CystoFluid"      "data/CystoFluid"       3               "true"           100
generate_yaml "Dca1"            "data/Dca1"             1               "false"          150
generate_yaml "Deepbacs"        "data/Deepbacs"         1               "false"          150
generate_yaml "Drive"           "data/Drive"            3               "true"           150
generate_yaml "DynamicNuclear"  "data/DynamicNuclear"   1               "false"          100
generate_yaml "EMSegmentation"  "data/EMSegmentation"   1               "false"          100
generate_yaml "FHPsAOP"         "data/FHPsAOP"          1               "false"          100
generate_yaml "Idrib"           "data/Idrib"            3               "true"           100
generate_yaml "Isic2016"        "data/Isic2016"         3               "true"           100
generate_yaml "Isic2018"        "data/Isic2018"         3               "true"           100
generate_yaml "Kvasir"          "data/Kvasir"           3               "true"           100
generate_yaml "M2caiSeg"        "data/M2caiSeg"         3               "true"           100
generate_yaml "MmWhsMr"         "data/MmWhsMr"          1               "false"          100
generate_yaml "Monusac"         "data/Monusac"          3               "true"           100
generate_yaml "MosMedPlus"      "data/MosMedPlus"       1               "false"          100
generate_yaml "Nuclei"          "data/Nuclei"           3               "true"           100
generate_yaml "Nuset"           "data/Nuset"            1               "false"          100
generate_yaml "Pandental"       "data/Pandental"        1               "false"          100
generate_yaml "Polyp"           "data/Polyp"            3               "true"           100
generate_yaml "PolypGen"        "data/PolypGen"         3               "true"           100
generate_yaml "Promise12"       "data/Promise12"        1               "false"          100
generate_yaml "RoboTool"        "data/RoboTool"         3               "true"           100
generate_yaml "TnbcNuclei"      "data/TnbcNuclei"       3               "true"           150
generate_yaml "UltrasoundNerve" "data/UltrasoundNerve"  1               "false"          100
generate_yaml "USforKidney"     "data/USforKidney"      1               "false"          100
generate_yaml "UwSkinCancer"    "data/UwSkinCancer"     3               "true"           150
generate_yaml "Wbc"             "data/Wbc"              3               "true"           100
generate_yaml "Yeaz"            "data/Yeaz"             1               "false"          100

echo "Generated ${output_directory}:"
ls -1 "${output_directory}"
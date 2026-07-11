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

    local lowercase_name
    lowercase_name=$(echo "$dataset_name" | tr '[:upper:]' '[:lower:]')

    cat > "${output_directory}/${lowercase_name}.yaml" << EOF
architecture:
  encoder_channels: [8, 16, 32]
  bottleneck_channels: 52
  kernel_size: 3
  normalization: "instance_norm"
  group_norm_num_groups: 8
  activation: h_swish
  upsampling_mode: nearest_neighbor
  use_residual_connections: false
  use_attention_gates: false
  dropout_probability: 0.0
  input_channels: ${input_channels}
  use_depthwise_separable_convolutions: false
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
  batch_size: 64
  epochs: 100
  dataset: ${dataset_name}
  image_size: 256
  # max_samples: 20000
  use_color_input: ${use_color_input}
  early_stopping_patience: 10
  early_stopping_minimum_improvement_delta: 0.001
  use_augmentation: true
  augmentation_apply_horizontal_flip: true
  augmentation_apply_vertical_flip: true
  augmentation_rotation_max_angle_degrees: 15
  hypothesis: "No hypothesis -  kaiming normal initialization."
  notes: "kaiming normal initialization."
EOF
}

#            dataset_name      data_root              input_channels  use_color_input
generate_yaml "Acdc"           "data/Acdc"            1               "false"
generate_yaml "Btcv"           "data/Btcv"            1               "false"
generate_yaml "CellNuclei"     "data/CellNuclei"      3               "true"
generate_yaml "Chaos"          "data/Chaos"           1               "false"
generate_yaml "EMSegmentation" "data/EMSegmentation"  1               "false"
generate_yaml "FHPsAOP"        "data/FHPsAOP"         1               "false"
generate_yaml "Isic2016"       "data/Isic2016"        3               "true"
generate_yaml "MmWhsMr"        "data/MmWhsMr"         1               "false"
generate_yaml "Nuset"          "data/Nuset"           1               "false"
generate_yaml "USforKidney"    "data/USforKidney"     1               "false"
generate_yaml "Wbc"            "data/Wbc"             3               "true"
generate_yaml "Yeaz"           "data/Yeaz"            1               "false"

echo "Generated ${output_directory}:"
ls -1 "${output_directory}"
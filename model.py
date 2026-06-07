import torch
import torch.nn as nn


def build_normalization_layer(normalization_type, num_channels, group_norm_num_groups):
    if normalization_type == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    elif normalization_type == "group_norm":
        return nn.GroupNorm(group_norm_num_groups, num_channels)
    elif normalization_type == "instance_norm":
        return nn.InstanceNorm2d(num_channels)
    else:
        return nn.Identity()


def build_activation_layer(activation_type):
    if activation_type == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif activation_type == "gelu":
        return nn.GELU()


class ConvolutionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, architecture_config):
        super().__init__()

        normalization_type = architecture_config["normalization"]
        group_norm_groups  = architecture_config["group_norm_num_groups"]
        activation_type    = architecture_config["activation"]
        kernel_size        = architecture_config["kernel_size"]
        same_size_padding  = kernel_size // 2

        self.apply_residual_connection = architecture_config["use_residual_connections"]

        self.double_convolution_with_norm_and_activation = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=same_size_padding, bias=False),
            build_normalization_layer(normalization_type, output_channels, group_norm_groups),
            build_activation_layer(activation_type),
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=same_size_padding, bias=False),
            build_normalization_layer(normalization_type, output_channels, group_norm_groups),
            build_activation_layer(activation_type),
        )

        if self.apply_residual_connection and input_channels != output_channels:
            self.channel_matching_shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        else:
            self.channel_matching_shortcut = nn.Identity()

    def forward(self, input_tensor):
        convolution_output = self.double_convolution_with_norm_and_activation(input_tensor)
        if self.apply_residual_connection:
            return convolution_output + self.channel_matching_shortcut(input_tensor)
        return convolution_output


class AttentionGate(nn.Module):
    def __init__(self, skip_channels, gate_channels, intermediate_channels=16):
        super().__init__()

        self.project_gating_signal_to_intermediate  = nn.Conv2d(gate_channels,  intermediate_channels, kernel_size=1, bias=False)
        self.project_skip_features_to_intermediate  = nn.Conv2d(skip_channels,  intermediate_channels, kernel_size=1, bias=False)
        self.collapse_intermediate_to_attention_map = nn.Conv2d(intermediate_channels, 1,               kernel_size=1, bias=False)

    def forward(self, skip_features, gating_signal):
        gating_projected = self.project_gating_signal_to_intermediate(gating_signal)

        gating_upsampled_to_skip_size = nn.functional.interpolate(
            gating_projected,
            size=skip_features.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        skip_projection             = self.project_skip_features_to_intermediate(skip_features)
        combined_projections        = torch.relu(skip_projection + gating_upsampled_to_skip_size)
        per_pixel_attention_weights = torch.sigmoid(self.collapse_intermediate_to_attention_map(combined_projections))

        return skip_features * per_pixel_attention_weights


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, architecture_config):
        super().__init__()
        self.convolution_block   = ConvolutionBlock(input_channels, output_channels, architecture_config)
        self.spatial_downsampler = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_tensor):
        skip_connection_features = self.convolution_block(input_tensor)
        downsampled_output       = self.spatial_downsampler(skip_connection_features)
        return downsampled_output, skip_connection_features


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, architecture_config):
        super().__init__()

        upsampling_mode = architecture_config["upsampling_mode"]

        if upsampling_mode == "transposed_conv":
            self.spatial_upsampler = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        else:
            self.spatial_upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
            )

        self.apply_attention_gating = architecture_config["use_attention_gates"]
        if self.apply_attention_gating:
            self.attention_gate = AttentionGate(
                skip_channels=output_channels,
                gate_channels=input_channels
            )

        channels_after_concatenating_upsampled_and_skip = 2 * output_channels
        self.convolution_block = ConvolutionBlock(channels_after_concatenating_upsampled_and_skip, output_channels, architecture_config)

        self.dropout = nn.Dropout2d(architecture_config["dropout_probability"]) \
            if architecture_config["dropout_probability"] > 0.0 else nn.Identity()

    def forward(self, input_tensor, skip_connection_tensor):
        upsampled_features = self.spatial_upsampler(input_tensor)

        if upsampled_features.shape[2:] != skip_connection_tensor.shape[2:]:
            upsampled_features = nn.functional.interpolate(
                upsampled_features,
                size=skip_connection_tensor.shape[2:],
                mode="bilinear",
                align_corners=True
            )

        if self.apply_attention_gating:
            skip_connection_tensor = self.attention_gate(skip_connection_tensor, input_tensor)

        concatenated_upsampled_and_skip = torch.cat([upsampled_features, skip_connection_tensor], dim=1)
        return self.dropout(self.convolution_block(concatenated_upsampled_and_skip))


class MicroUNet(nn.Module):
    def __init__(self, architecture_config, output_channels=1):
        super().__init__()

        encoder_channel_sizes   = architecture_config["encoder_channels"]
        bottleneck_channel_size = architecture_config["bottleneck_channels"]
        input_channels          = architecture_config.get("input_channels", 1)

        self.encoder_blocks = nn.ModuleList()
        current_channel_count = input_channels
        for encoder_output_channels in encoder_channel_sizes:
            self.encoder_blocks.append(EncoderBlock(current_channel_count, encoder_output_channels, architecture_config))
            current_channel_count = encoder_output_channels

        self.bottleneck_convolution_block = ConvolutionBlock(current_channel_count, bottleneck_channel_size, architecture_config)

        self.decoder_blocks = nn.ModuleList()
        decoder_input_channel_count = bottleneck_channel_size
        for decoder_output_channels in reversed(encoder_channel_sizes):
            self.decoder_blocks.append(DecoderBlock(decoder_input_channel_count, decoder_output_channels, architecture_config))
            decoder_input_channel_count = decoder_output_channels

        self.final_one_by_one_segmentation_head = nn.Conv2d(decoder_input_channel_count, output_channels, kernel_size=1)

    def forward(self, input_tensor):
        input_spatial_size   = input_tensor.shape[2:]
        all_skip_connections = []
        current_feature_map  = input_tensor

        for encoder_block in self.encoder_blocks:
            current_feature_map, skip_features = encoder_block(current_feature_map)
            all_skip_connections.append(skip_features)

        current_feature_map = self.bottleneck_convolution_block(current_feature_map)

        for decoder_block, skip_features in zip(self.decoder_blocks, reversed(all_skip_connections)):
            current_feature_map = decoder_block(current_feature_map, skip_features)

        raw_output = self.final_one_by_one_segmentation_head(current_feature_map)

        if raw_output.shape[2:] != input_spatial_size:
            return nn.functional.interpolate(raw_output, size=input_spatial_size, mode="bilinear", align_corners=True)

        return raw_output

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(architecture_config, device):
    model                      = MicroUNet(architecture_config)
    total_trainable_parameters = model.count_trainable_parameters()
    print(f"Model: MicroUNet | Parameters: {total_trainable_parameters:,} ({total_trainable_parameters / 1e6:.4f}M)")
    assert total_trainable_parameters < 100_000, f"Model exceeds 0.1M parameter limit: {total_trainable_parameters:,}"
    return model.to(device)
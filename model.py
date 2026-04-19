import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks — each one is constructed from yaml architecture constants
# ---------------------------------------------------------------------------

def build_normalization_layer(normalization, num_channels, group_norm_num_groups):
    """
    Normalization sits between Conv and activation.
    All three operate on the same channel count — normalization does not change shape.

    batch_norm    — normalizes across the batch. Unstable with small batches.
    group_norm    — normalizes within channel groups. Batch-size independent.
                    Better for small medical imaging batches.
    instance_norm — normalizes each sample independently per channel.
    """
    if normalization == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    elif normalization == "group_norm":
        return nn.GroupNorm(group_norm_num_groups, num_channels)
    elif normalization == "instance_norm":
        return nn.InstanceNorm2d(num_channels)
    else:
        return nn.Identity()


def build_activation_layer(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif activation == "gelu":
        return nn.GELU()


class ConvolutionBlock(nn.Module):
    """
    Two rounds of: Conv -> Norm -> Activation

    Each conv uses kernel=3, padding=1, which keeps spatial size identical:
        output_size = (H - 3 + 2*1) / 1 + 1 = H

    If use_residual is True, the input is added back to the output after both convolutions.
    The block then only needs to learn what to *add* (the residual / the edit),
    rather than the full transformation from scratch — this keeps gradients
    flowing through the shortcut even in deeper networks.

    If input and output channels differ, a 1x1 conv aligns the shortcut channels
    before adding.
    """

    def __init__(self, input_channels, output_channels, architecture_config):
        super().__init__()

        normalization   = architecture_config["normalization"]
        groups          = architecture_config["group_norm_num_groups"]
        activation      = architecture_config["activation"]
        self.use_residual = architecture_config["use_residual_connections"]

        self.double_convolution = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            build_normalization_layer(normalization, output_channels, groups),
            build_activation_layer(activation),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            build_normalization_layer(normalization, output_channels, groups),
            build_activation_layer(activation),
        )

        # Shortcut: if channel counts differ, 1x1 conv matches them before addition
        if self.use_residual and input_channels != output_channels:
            self.shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, input_tensor):
        convolved = self.double_convolution(input_tensor)
        if self.use_residual:
            # output = what to add + original input — block only learns the difference
            return convolved + self.shortcut(input_tensor)
        return convolved


class AttentionGate(nn.Module):
    """
    Filters skip connection features before they reach the decoder concat.
    Instead of passing all skip features blindly, it suppresses irrelevant
    background and amplifies regions likely to contain the glottis.

    Takes two inputs:
      skip_features  — sharp spatial detail from encoder  e.g. [B, 32, 64, 64]
      gating_signal  — semantic features from decoder below e.g. [B, 64, 32, 32]

    The gating signal is from one level deeper → it has more channels (U-Net doubles
    channels at each encoder level) and smaller spatial size.

    Step 1 — reduce channels on the smaller map first (cheaper), then upsample:
              Conv(64→16) on [B,64,32,32] → [B,16,32,32] → upsample → [B,16,64,64]
              Conv(32→16) on [B,32,64,64] → [B,16,64,64]
              Now both are the same shape and can be added.

    Step 2 — combine, add non-linearity, collapse to one weight per pixel:
              ReLU(skip_proj + gate_proj) → Conv(16→1) → sigmoid → [B,1,64,64]
              Each value 0→1: high = relevant pixel, low = background

    Step 3 — broadcast attention map across all skip channels:
              skip_features [B,32,64,64] × attention_map [B,1,64,64] → [B,32,64,64]
    """

    def __init__(self, skip_channels, gate_channels, intermediate_channels=16):
        super().__init__()

        # Conv first on the smaller gating signal (32x32), then upsample to skip size (64x64)
        # Doing conv on the smaller map is cheaper than upsampling first and then running conv
        self.gate_projection    = nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=False)
        self.skip_projection    = nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=False)
        self.attention_collapse = nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False)
        self.upsample_gate      = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, skip_features, gating_signal):
        gate_projected = self.upsample_gate(self.gate_projection(gating_signal))  # [B, 16, 64, 64]
        skip_projected = self.skip_projection(skip_features)                       # [B, 16, 64, 64]

        combined       = torch.relu(skip_projected + gate_projected)               # [B, 16, 64, 64]
        attention_map  = torch.sigmoid(self.attention_collapse(combined))          # [B,  1, 64, 64]

        # [B,1,64,64] broadcasts across all skip channels — relevant pixels survive, background suppressed
        return skip_features * attention_map


class EncoderBlock(nn.Module):
    """
    One level of the left side of the U.

    ConvolutionBlock learns features at this resolution — output saved as skip connection
    because it still has sharp, precise spatial detail before pooling blurs it.

    MaxPool2d halves spatial size (256→128→64...) which expands the receptive field:
    each pixel after pooling represents a 2x2 region of the original, so the next
    conv layer sees a larger patch of the image without any extra parameters.
    """

    def __init__(self, input_channels, output_channels, architecture_config):
        super().__init__()
        self.convolution_block = ConvolutionBlock(input_channels, output_channels, architecture_config)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_tensor):
        # Save before pooling — these are the sharp "where" features the decoder needs via skip connection
        skip_connection_features = self.convolution_block(input_tensor)
        downsampled_features     = self.pooling(skip_connection_features)
        return downsampled_features, skip_connection_features


class DecoderBlock(nn.Module):
    """
    One level of the right side of the U.

    Step 1 — Upsample: doubles spatial size
             Result is blurry but semantically rich ("glottis is roughly here")

    Step 2 — Optionally filter the skip connection through an attention gate
             before concatenating (suppresses irrelevant background pixels)

    Step 3 — Concatenate with skip connection: stacked along dim=1
             [upsampled channels | skip channels] — spatial grids must match, channels just stack
             Result: sharp "where" (from skip) fused with semantic "what" (from upsample)

    Step 4 — ConvolutionBlock cleans up the concatenated features and halves channels back
    """

    def __init__(self, input_channels, output_channels, architecture_config):
        super().__init__()

        upsampling_mode = architecture_config["upsampling_mode"]

        if upsampling_mode == "transposed_conv":
            # Learned upsampling — network decides how to fill in the gaps
            self.upsampling = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        else:
            # Fixed bilinear interpolation — no parameters, smoother result
            # Followed by 1x1 conv to reduce channels since bilinear doesn't do that
            self.upsampling = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
            )

        self.use_attention = architecture_config["use_attention_gates"]
        if self.use_attention:
            # skip has output_channels, gating signal (input to this block) has input_channels
            self.attention_gate = AttentionGate(
                skip_channels=output_channels,
                gate_channels=input_channels
            )

        # After concat: output_channels (upsampled) + output_channels (skip) = input_channels
        self.convolution_block = ConvolutionBlock(input_channels, output_channels, architecture_config)

        self.dropout = nn.Dropout2d(architecture_config["dropout_probability"]) \
            if architecture_config["dropout_probability"] > 0.0 else nn.Identity()

    def forward(self, input_tensor, skip_connection_tensor):
        upsampled = self.upsampling(input_tensor)

        if self.use_attention:
            # Gate filters skip features using the semantic knowledge from input_tensor
            skip_connection_tensor = self.attention_gate(skip_connection_tensor, input_tensor)

        # Both tensors: same H x W. Channels stack along dim=1.
        concatenated = torch.cat([upsampled, skip_connection_tensor], dim=1)
        return self.dropout(self.convolution_block(concatenated))


# ---------------------------------------------------------------------------
# Full U-Net — built entirely from yaml architecture config
# ---------------------------------------------------------------------------

class MicroUNet(nn.Module):
    """
    U-Net under 0.1M parameters for binary glottis segmentation.

    Encoder progressively downsamples — builds semantic understanding, expands receptive field.
    Decoder progressively upsamples — reconstructs pixel-precise locations via skip connections.

    Depth and channel sizes are driven entirely by architecture_config["encoder_channels"].
    e.g. encoder_channels: [16, 32] gives depth=2 with those channel widths.

    Final 1x1 conv: collapses last decoder channels to 1 value per pixel (raw logit).
    Sigmoid is NOT applied here — BCEWithLogitsLoss in train.py handles that
    numerically more stably than applying sigmoid manually.
    """

    def __init__(self, architecture_config, input_channels=1, output_channels=1):
        super().__init__()

        encoder_channels    = architecture_config["encoder_channels"]
        bottleneck_channels = architecture_config["bottleneck_channels"]

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        current_channels = input_channels
        for out_channels in encoder_channels:
            self.encoder_blocks.append(EncoderBlock(current_channels, out_channels, architecture_config))
            current_channels = out_channels

        # --- Bottleneck: smallest spatial map, richest semantic understanding ---
        self.bottleneck = ConvolutionBlock(current_channels, bottleneck_channels, architecture_config)

        # --- Decoder: mirror of encoder, channels in reverse ---
        self.decoder_blocks = nn.ModuleList()
        decoder_in_channels = bottleneck_channels
        for out_channels in reversed(encoder_channels):
            self.decoder_blocks.append(DecoderBlock(decoder_in_channels, out_channels, architecture_config))
            decoder_in_channels = out_channels

        # 1x1 conv: collapses final feature channels to one segmentation mask per pixel
        self.final_segmentation_head = nn.Conv2d(decoder_in_channels, output_channels, kernel_size=1)

    def forward(self, input_tensor):
        # Encoder: downsample and save skip connections at each level
        skip_connections = []
        current = input_tensor
        for encoder_block in self.encoder_blocks:
            current, skip = encoder_block(current)
            skip_connections.append(skip)

        current = self.bottleneck(current)

        # Decoder: upsample and fuse with skip connections in reverse order
        # skip_connections[-1] is the shallowest skip (sharpest, largest spatial map)
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            current = decoder_block(current, skip)

        # One raw logit per pixel — sigmoid applied inside the loss function
        return self.final_segmentation_head(current)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(architecture_config, device):
    model = MicroUNet(architecture_config)
    parameter_count = model.count_parameters()
    print(f"Model: MicroUNet | Parameters: {parameter_count:,} ({parameter_count / 1e6:.4f}M)")
    assert parameter_count < 100_000, f"Model exceeds 0.1M parameter limit: {parameter_count:,}"
    return model.to(device)

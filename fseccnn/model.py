# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A standard convolutional block with two convolutions, normalization, activation,
    and optional dropout. Optionally includes a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 norm_layer=nn.BatchNorm1d, activation_fn=nn.ReLU, use_residual=True, dropout_p=0.0):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                               bias=False if norm_layer else True)
        self.norm1 = norm_layer(out_channels) if norm_layer else nn.Identity()
        # self.act1 = activation_fn(inplace=True) # <<< POTENTIAL ISSUE
        self.act1 = activation_fn()  # <<< CORRECTED: Remove inplace=True or set to False if option exists
        self.dropout1 = nn.Dropout1d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False if norm_layer else True)
        self.norm2 = norm_layer(out_channels) if norm_layer else nn.Identity()
        # self.act2 = activation_fn(inplace=True) # <<< POTENTIAL ISSUE
        self.act2 = activation_fn()  # <<< CORRECTED: Remove inplace=True
        self.dropout2 = nn.Dropout1d(dropout_p) if dropout_p > 0 else nn.Identity()

        if self.use_residual and in_channels == out_channels and stride == 1:
            self.residual_projection = nn.Identity()
        elif self.use_residual:
            self.residual_projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels) if norm_layer else nn.Identity()
            )
        else:
            self.residual_projection = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        if self.use_residual:
            # Ensure residual projection output matches 'out' for addition
            projected_residual = self.residual_projection(residual)
            out += projected_residual

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=2, pool_stride=2, **conv_block_kwargs):
        super().__init__()
        # ConvBlock will receive dropout_p through **conv_block_kwargs
        self.conv_block = ConvBlock(in_channels, out_channels, **conv_block_kwargs)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride) if pool_kernel > 0 else nn.Identity()

    def forward(self, x):
        skip_connection = self.conv_block(x)
        pooled_output = self.pool(skip_connection)
        return pooled_output, skip_connection


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 upsample_kernel=2, upsample_stride=2, **conv_block_kwargs):
        super().__init__()
        # ConvTranspose1d typically doesn't have dropout directly applied to it.
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels,
                                         kernel_size=upsample_kernel, stride=upsample_stride)
        # The conv_block input will be out_channels from upconv + skip_channels
        # ConvBlock will receive dropout_p through **conv_block_kwargs
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels, **conv_block_kwargs)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        if x.shape[2] != skip_connection.shape[2]:
            diff = skip_connection.shape[2] - x.shape[2]
            if diff > 0:
                padding_left = diff // 2
                padding_right = diff - padding_left
                x = F.pad(x, (padding_left, padding_right))
            elif diff < 0:
                abs_diff = abs(diff)
                crop_left = abs_diff // 2
                crop_right = abs_diff - crop_left
                x = x[:, :, crop_left: x.shape[2] - crop_right]
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 initial_filters=64, depth=4, kernel_size=3,
                 pool_kernel=2, pool_stride=2,
                 norm_layer=nn.BatchNorm1d, activation_fn_class=nn.ReLU, # Changed to activation_fn_class
                 use_residual_conv=True, output_activation_class=nn.Sigmoid, # Changed to output_activation_class
                 dropout_p=0.0):
        super().__init__()
        # ... (parameter validation) ...
        if not (isinstance(initial_filters, int) and initial_filters > 0):
            raise ValueError("initial_filters must be a positive integer.")
        if not (isinstance(depth, int) and depth >= 1):
            raise ValueError("depth must be a positive integer >= 1.")
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError("dropout_p must be between 0.0 and 1.0 (exclusive of 1.0).")

        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        common_conv_kwargs = {
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "norm_layer": norm_layer,
            "activation_fn": activation_fn_class, # Pass the class
            "use_residual": use_residual_conv,
            "dropout_p": dropout_p
        }
        # ... (rest of UNet1D __init__ is the same) ...
        current_channels = in_channels
        for i in range(depth):
            filters_out = initial_filters * (2**i)
            self.encoders.append(
                EncoderBlock(current_channels, filters_out, pool_kernel, pool_stride, **common_conv_kwargs)
            )
            current_channels = filters_out
        bottleneck_filters_in = initial_filters * (2**(depth - 1))
        bottleneck_filters_out = initial_filters * (2**depth)
        self.bottleneck = ConvBlock(bottleneck_filters_in, bottleneck_filters_out, **common_conv_kwargs)
        for i in range(depth -1, -1, -1):
            decoder_filters_in = initial_filters * (2**(i + 1))
            skip_conn_filters = initial_filters * (2**i)
            decoder_filters_out = initial_filters * (2**i)
            self.decoders.append(
                DecoderBlock(decoder_filters_in, skip_conn_filters, decoder_filters_out,
                             upsample_kernel=pool_kernel, upsample_stride=pool_stride,
                             **common_conv_kwargs)
            )
        self.out_conv = nn.Conv1d(initial_filters, out_channels, kernel_size=1)
        # Instantiate the output activation here
        self.output_activation = output_activation_class() if output_activation_class else nn.Identity()


    def forward(self, x): # Forward pass remains the same
        if x.ndim == 2: x = x.unsqueeze(1)
        skip_connections = []
        for i in range(self.depth):
            x, skip = self.encoders[i](x)
            skip_connections.append(skip)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(self.depth):
            x = self.decoders[i](x, skip_connections[i])
        x = self.out_conv(x)
        x = self.output_activation(x)
        return x


# --- Example Usage (for testing the model) ---
if __name__ == '__main__':
    print("Testing UNet1D with Dropout...")

    # Configuration 1: Default with dropout
    model1 = UNet1D(in_channels=1, out_channels=1, initial_filters=16, depth=3, dropout_p=0.1)
    model1.train()  # Dropout is active during training
    test_input1 = torch.randn(4, 256)
    output1 = model1(test_input1)
    print(f"Model 1 (depth=3, filters=16, dropout=0.1) input: {test_input1.shape}, output: {output1.shape}")
    assert output1.shape == (4, 1, 256)
    model1.eval()  # Dropout is inactive during evaluation
    output1_eval = model1(test_input1)
    # Check that outputs are different if dropout was active (probabilistic)
    # This is not a perfect check due to randomness, but a sanity check.
    # if dropout_p > 0 and model1.training was True before eval:
    #     assert not torch.allclose(output1, output1_eval) # This might fail due to seed or if dropout by chance does nothing

    # Configuration 2: No dropout
    model2 = UNet1D(in_channels=1, out_channels=1, initial_filters=16, depth=3, dropout_p=0.0)
    model2.train()
    output2_train = model2(test_input1)
    model2.eval()
    output2_eval = model2(test_input1)
    print(f"Model 2 (dropout=0.0) train_out_shape: {output2_train.shape}, eval_out_shape: {output2_eval.shape}")
    assert torch.allclose(output2_train, output2_eval)  # Should be same if no dropout

    # Configuration 3: Deeper, more filters, dropout
    model3 = UNet1D(in_channels=1, out_channels=1, initial_filters=32, depth=4, kernel_size=3, dropout_p=0.2)
    test_input3 = torch.randn(2, 512)
    output3 = model3(test_input3)
    print(f"Model 3 (depth=4, filters=32, dropout=0.2) input: {test_input3.shape}, output: {output3.shape}")
    assert output3.shape == (2, 1, 512)

    print("UNet1D with dropout tests passed basic shape checks.")

    from torchinfo import summary

    print("\nModel 1 (with dropout) Summary:")
    summary(model1, input_size=(4, 1, 256))
"""
Custom neural network architectures for Microscope game
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MicroscopeCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the Microscope game.

    Designed for 7x7x4 board (7x7 grid with 4 channels: Green pieces, Blue pieces, Turn indicator, Selected piece).
    Uses residual connections and spatial features to learn board patterns.

    Architecture:
    - Input: 7x7x4 board
    - Conv layers with increasing filters (extract patterns)
    - Residual connections (help training deeper networks)
    - Global average pooling (positional invariance)
    - Dense layers for final feature representation
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        """
        Args:
            observation_space: The observation space (should be 7x7x4)
            features_dim: Number of output features (default 256)
        """
        super().__init__(observation_space, features_dim)

        # Input channels: 2 (Blue and Green)
        n_input_channels = observation_space.shape[2]

        # Convolutional layers designed for small 7x7 board
        # Use small 3x3 kernels to preserve spatial information
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Residual connection from conv1 to conv3
        self.residual = nn.Conv2d(32, 128, kernel_size=1)

        # Additional conv layer for deeper feature extraction
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense layers for final features
        # 128 from conv + 98 from flattened board state
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, features_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            observations: Board state tensor of shape (batch, 7, 7, 4)
                         Channels: 0=Green, 1=Blue, 2=Turn, 3=Selected piece

        Returns:
            Feature tensor of shape (batch, features_dim)
        """
        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        x = observations.permute(0, 3, 1, 2).float()

        # First conv block
        x1 = self.activation(self.bn1(self.conv1(x)))

        # Second conv block
        x = self.activation(self.bn2(self.conv2(x1)))

        # Third conv block with residual
        x = self.activation(self.bn3(self.conv3(x)))
        residual = self.residual(x1)
        x = x + residual  # Residual connection

        # Fourth conv block
        x = self.activation(self.bn4(self.conv4(x)))

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dense layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class LightweightMicroscopeCNN(BaseFeaturesExtractor):
    """
    Lightweight CNN for faster training with fewer parameters.
    Good for initial experimentation or when compute is limited.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]

        # Simpler architecture
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, features_dim)

        self.activation = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.permute(0, 3, 1, 2).float()

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DeepMicroscopeCNN(BaseFeaturesExtractor):
    """
    Deep residual CNN for maximum performance.
    Uses multiple residual blocks similar to ResNet architecture.
    Best for final training when you have compute resources.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]

        # Initial convolution
        self.conv_input = nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)

        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128)
        self.res_block2 = self._make_residual_block(128, 256)
        self.res_block3 = self._make_residual_block(256, 256)

        # Policy and value heads preparation
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.value_conv = nn.Conv2d(256, 64, kernel_size=1)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final features
        self.fc = nn.Linear(128 + 64, features_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block with two conv layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.permute(0, 3, 1, 2).float()

        # Initial conv
        x = self.activation(self.bn_input(self.conv_input(x)))

        # Residual blocks
        identity = x
        x = self.res_block1(x)
        if identity.size(1) != x.size(1):
            identity = nn.Conv2d(identity.size(1), x.size(1), kernel_size=1).to(x.device)(identity)
        x = self.activation(x + identity)

        identity = x
        x = self.res_block2(x)
        if identity.size(1) != x.size(1):
            identity = nn.Conv2d(identity.size(1), x.size(1), kernel_size=1).to(x.device)(identity)
        x = self.activation(x + identity)

        identity = x
        x = self.res_block3(x)
        x = self.activation(x + identity)

        # Split into policy and value streams
        policy_features = self.global_pool(self.activation(self.policy_conv(x)))
        value_features = self.global_pool(self.activation(self.value_conv(x)))

        # Concatenate and flatten
        x = torch.cat([policy_features, value_features], dim=1)
        x = x.view(x.size(0), -1)

        # Final features
        x = self.dropout(x)
        x = self.fc(x)

        return x

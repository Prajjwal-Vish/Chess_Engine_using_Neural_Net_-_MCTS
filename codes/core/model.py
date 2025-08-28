import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    """
    The definitive AlphaZero-style neural network architecture.
    """
    def __init__(self, num_residual_blocks, policy_size, input_channels=24):
        super(AlphaZeroNet, self).__init__()
        
        # --- The "Body" or "Tower" of the Network ---
        self.initial_conv = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(256)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(256) for _ in range(num_residual_blocks)]
        )
        
        # --- The "Heads" of the Network ---

        # 1. The Value Head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8*8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # 2. The Policy Head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_size)

    def forward(self, x):
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks:
            out = block(out)
            
        # --- Value Head Path ---
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 8*8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        # --- Policy Head Path ---
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)
        
        return value, policy

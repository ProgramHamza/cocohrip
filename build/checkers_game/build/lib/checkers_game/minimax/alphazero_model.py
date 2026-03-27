import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 8
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # from_r, from_c, to_r, to_c flattened


def action_index(fr: int, fc: int, tr: int, tc: int) -> int:
    return ((fr * BOARD_SIZE + fc) * BOARD_SIZE + tr) * BOARD_SIZE + tc


def index_to_action(idx: int):
    tc = idx % BOARD_SIZE
    idx //= BOARD_SIZE
    tr = idx % BOARD_SIZE
    idx //= BOARD_SIZE
    fc = idx % BOARD_SIZE
    idx //= BOARD_SIZE
    fr = idx
    return fr, fc, tr, tc


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return F.relu(out)


class AZNet(nn.Module):
    def __init__(self, channels: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(5, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            out = block(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy_logits, value


class AZTrainer:
    def __init__(self, model: AZNet, lr: float = 1e-3, weight_decay: float = 1e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(self, states: torch.Tensor, policy_targets: torch.Tensor, value_targets: torch.Tensor) -> float:
        self.model.train()
        policy_logits, values = self.model(states)
        policy_loss = -torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

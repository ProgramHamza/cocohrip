import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class StateActionQNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def save(self, file_name: str = "rl_model.pth") -> None:
        folder = os.path.join(".", "model")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name)
        torch.save(self.state_dict(), path)


class DQNTrainer:
    def __init__(
        self,
        model: StateActionQNet,
        lr: float,
        gamma: float,
        optimizer: optim.Optimizer | None = None,
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.gamma = gamma
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scheduler = scheduler

    def train_step(
        self,
        states: List,
        actions: List,
        rewards: List,
        next_states: List,
        dones: List[bool],
        next_actions_list: List[List],
        target_model: StateActionQNet,
        max_grad_norm: float = 1.0,
    ) -> None:
        state_tensor = torch.tensor(states, dtype=torch.float)
        action_tensor = torch.tensor(actions, dtype=torch.float)
        reward_tensor = torch.tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float)
        done_tensor = torch.tensor(dones, dtype=torch.bool)

        preds = self.model(state_tensor, action_tensor).view(-1)
        targets = reward_tensor.clone().view(-1)

        for idx, (done, next_actions) in enumerate(zip(done_tensor, next_actions_list)):
            if done or not next_actions:
                continue
            ns = next_state_tensor[idx].unsqueeze(0).repeat(len(next_actions), 1)
            na = torch.tensor(next_actions, dtype=torch.float)
            # Double DQN: select action with online net, evaluate with target net
            with torch.no_grad():
                q_online = self.model(ns, na).squeeze()
                best_idx = torch.argmax(q_online).item()
                q_next = target_model(ns[best_idx].unsqueeze(0), na[best_idx].unsqueeze(0)).squeeze()
            targets[idx] = reward_tensor[idx] + self.gamma * q_next.detach()

        self.optimizer.zero_grad()
        loss = self.criterion(preds, targets)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

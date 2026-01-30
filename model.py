import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="models/model.pth"):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="models/model.pth", device=None):
        if device is None:
            device = torch.device("cpu")
        state = torch.load(file_name, map_location=device)
        self.load_state_dict(state)
        self.to(device)


class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        done_tensor = torch.tensor(done, dtype=torch.bool).to(self.device)
        with torch.no_grad():
            max_next_q = self.model(next_state).max(1).values
            q_new = reward + self.gamma * max_next_q * (~done_tensor)

        action_idx = torch.argmax(action, dim=1)
        target[torch.arange(target.size(0)), action_idx] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

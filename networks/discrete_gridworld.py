import torch.nn as nn
import torch
from misc.utils import to_onehot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhaseI(nn.Module):
    def __init__(self, obs, n_actions, cumulants, tasks):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(PhaseI, self).__init__()
        self.num_cumulants = cumulants
        self.num_tasks = tasks
        self.n_actions = n_actions
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        n = obs[1]
        m = obs[2]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.psi = nn.Sequential(
            nn.Linear(self.image_embedding_size + 1 + self.num_tasks, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_cumulants * self.n_actions),
        )
        self.phi = nn.Sequential(
            nn.Linear(self.image_embedding_size + 1 + self.num_tasks, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_cumulants * self.n_actions),
        )

        self.w = nn.Parameter(torch.randn(self.num_tasks, self.num_cumulants))

    def forward(self, x, color, battery_level):
        features = self.features(x)
        features = features.reshape(x.shape[0], -1)

        if x.shape[0] == 1:
            battery_level = torch.tensor(battery_level).unsqueeze(-1).unsqueeze(-1)
            color = torch.tensor(color).unsqueeze(-1)

        color_encoding = to_onehot(color, self.num_tasks)
        features = torch.cat([features, battery_level.to(device), color_encoding.to(device)], dim=-1)

        output_psi = self.psi(features)
        output_psi = output_psi.view(x.shape[0], self.num_cumulants, self.n_actions)
        assistive_q = torch.einsum("bca, tc  -> bta", output_psi, self.w)
        assistive_q = assistive_q[torch.arange(x.shape[0]), color.squeeze(), :]

        output_phi = self.phi(features)
        output_phi = output_phi.view(x.shape[0], self.num_cumulants, self.n_actions)
        assistive_rewards = torch.einsum("bca, tc  -> bta", output_phi, self.w)
        assistive_rewards = assistive_rewards[torch.arange(x.shape[0]), color.squeeze(), :]

        return assistive_q, assistive_rewards

class PhaseII(nn.Module):
    def __init__(self, obs, n_actions, cumulants, tasks):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(PhaseII, self).__init__()
        self.num_cumulants = cumulants
        self.num_tasks = tasks
        self.n_actions = n_actions
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        n = obs[1]
        m = obs[2]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.psi = nn.Sequential(
            nn.Linear(self.image_embedding_size + 1 + self.num_tasks, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_cumulants * self.n_actions),
        )
        self.phi = nn.Sequential(
            nn.Linear(self.image_embedding_size + 1 + self.num_tasks, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_cumulants * self.n_actions),
        )

        self.w = nn.Parameter(torch.randn(self.num_cumulants))

    def forward(self, x, color, battery_level):
        features = self.features(x)
        features = features.reshape(x.shape[0], -1)

        if x.shape[0] == 1:
            battery_level = torch.tensor(battery_level).unsqueeze(-1).unsqueeze(-1)
            color = torch.tensor(color).unsqueeze(-1)

        color_encoding = to_onehot(color, self.num_tasks)
        features = torch.cat([features, battery_level.to(device), color_encoding.to(device)], dim=-1)

        output_psi = self.psi(features)
        output_psi = output_psi.view(x.shape[0], self.num_cumulants, self.n_actions)
        assistive_q = torch.einsum("bca, c  -> ba", output_psi, self.w)

        output_phi = self.phi(features)
        output_phi = output_phi.view(x.shape[0], self.num_cumulants, self.n_actions)
        assistive_rewards = torch.einsum("bca, c  -> ba", output_phi, self.w)

        return assistive_q, assistive_rewards



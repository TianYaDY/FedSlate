from torch import nn
from torch.nn.utils import weight_norm





class QNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, output_dim),
            nn.Tanh()
        )

        self.target = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, 4096),
            nn.Mish(),
            nn.Linear(4096, output_dim),
            nn.Tanh()
        )

        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, inputs, model):
        if model == "online":
            inputs = inputs
            return self.online(inputs)
        elif model == "target":
            inputs = inputs
            return self.target(inputs)


class MLPNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.Mish(),
            nn.Linear(2048, 2048),
            nn.Mish(),
            nn.Linear(2048, 2048),
            nn.Mish(),
            nn.Linear(2048, 2048),
            nn.Mish(),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

        self.target = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.Mish(),
            nn.Linear(2048, 2048),
            nn.Mish(),
            nn.Linear(2048, 2048),
            nn.Mish(),
            nn.Linear(2048, 2048),
            nn.Mish(),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, inputs, model):
        if model == "online":
            return self.online(inputs)
        elif model == "target":
            return self.target(inputs)

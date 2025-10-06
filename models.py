import torch
import torchvision


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, activation_layer):
        assert len(hidden_channels) > 0
        super().__init__()
        self.mlp = torchvision.ops.MLP(
            in_channels, hidden_channels, activation_layer=activation_layer
        )
        self.in_features = in_channels
        self.out_features = hidden_channels[-1]
        if self.out_features == 1:
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, x):
        logits = self.mlp(x.view(x.shape[0], -1))
        probs = logits
        return probs


def make_mlp(
    in_channels,
    hidden_channels,
    activation_layer=torch.nn.modules.activation.LeakyReLU,
):
    return MLP(in_channels, hidden_channels, activation_layer)


class MLPRegressor(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_epochs,
        lr,
        activation_layer=torch.nn.modules.activation.LeakyReLU,
        device="cuda",
    ):
        super().__init__()
        self.in_channels = None
        self.mlp = None
        self.hidden_channels = hidden_channels
        self.activation_layer = activation_layer
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device

    def forward(self, x):
        return self.predict(x, to_numpy=False)

    def predict(self, x, to_numpy=True):
        assert (
            self.mlp is not None
        ), "MLPRegressor must be fitted before predict"
        self.mlp.eval()
        x = torch.Tensor(x).to(self.device)
        pred = self.mlp(x).squeeze(-1)
        if to_numpy:
            return pred.cpu().detach().numpy()
        else:
            return pred

    def fit(self, x, y):
        x = torch.Tensor(x).to(self.device)
        y = torch.Tensor(y).to(self.device).unsqueeze(-1)

        if self.in_channels is None:
            self.in_channels = x.shape[1]
            self.mlp = make_mlp(
                self.in_channels, self.hidden_channels, self.activation_layer
            )
            self.mlp.to(self.device)

        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()

        self.mlp.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.mlp(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        return self

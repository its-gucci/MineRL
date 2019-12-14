import torch
import torch.nn as nn
import torch.nn.functional as F

class GNetwork(nn.Module):
  def __init__(self, embed_dim, input_shape=(3, 64, 64), k=1):
    super(GNetwork, self).__init__()
    self.conv1 = nn.Conv2d(2*input_shape[0], 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.linear = nn.Linear(64 * 6 * 6, embed_dim * k)

  def forward(self, s1, s2):
    x = torch.cat([s1, s2], axis=1)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.linear(x.reshape((-1, 64*6*6)))
    return x

class FNetwork(nn.Module):
  def __init__(self, n_action, embed_dim, k=1):
    super(FNetwork, self).__init__()
    self.embedding = nn.Linear(embed_dim * k, n_action * k)

  def forward(self, x):
    return self.embedding(x)

  def get_action(self, x):
    action = F.softmax(self.embedding(x), dim=-1)
    return action

class PolicyNet(nn.Module):
  def __init__(self, embed_dim, input_shape=(3, 64, 64)):
    super(PolicyNet, self).__init__()
    self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.linear = nn.Linear(1024, 2*embed_dim)

    self.embed_dim = embed_dim

    self.trainable_params = list(self.parameters())

    self.param_sizes = [p.data.numpy().size for p in self.trainable_params]

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.linear(x.reshape(-1,1024))
    a, b = torch.split(x, self.embed_dim, dim=-1)
    return torch.distributions.normal.Normal(a, F.softplus(b))

  def get_params(self):
    params = np.concatenate([p.contiguous().view(-1).data.numpy() for p in self.trainable_params])
    return params.copy()

  def set_params(self):
    current_idx = 0
    for idx, param in enumerate(self.trainable_params):
      vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
      vals = vals.reshape(self.param_shapes[idx])
      param.data = torch.from_numpy(vals).float()
      current_idx += self.param_sizes[idx]



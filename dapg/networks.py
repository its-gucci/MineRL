import torch
import torch.nn as nn
import torch.nn.functional as functional

class GNetwork(nn.Module):
	def __init__(self, embed_dim, input_shape=(3, 64, 64), k=1):
		super(GNetwork, self).__init__()
		self.conv1 = nn.Conv2d(2*input_shape[0], 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
		self.linear = nn.Linear(32, embed_dim * k)

	def forward(self, s1, s2):
		x = torch.cat([s1, s2], axis=1)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.linear(x)
		return x

class FNetwork(nn.Module):
	def __init__(self, n_action, embed_dim, k=1):
		super(FNetwork, self).__init__()
		self.embedding(embed_dim * k, n_action * k)

	def forward(self, x):
		return self.embedding(x)

	def get_action(self, x):
		action = F.softmax(self.embedding(x), dim=-1)
		return action

class PolicyNet(nn.Module):
	def __init__(self, embed_dim, input_channels=3):
		super(BCNet, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.linear = nn.Linear(64, embed_dim)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.linear(x)
		return x


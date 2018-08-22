import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Image 112
        self.img_size = 82
        self.hidden = 128

        # Encode
        # self.fc1 = nn.Linear(self.img_size * self.img_size, self.hidden)
        # self.fc21 = nn.Linear(self.hidden, 8)  # mu
        # self.fc22 = nn.Linear(self.hidden, 8)  # logvar

        self.fc1 = nn.Linear(self.img_size * self.img_size, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 256)
        self.fc31 = nn.Linear(256, 128)  # mu
        self.fc32 = nn.Linear(256, 128)  # logvar

        # Decode
        # self.fc3 = nn.Linear(8, self.hidden)
        # self.fc4 = nn.Linear(self.hidden, self.img_size * self.img_size)
        self.fc3 = nn.Linear(128, self.hidden)
        self.fc4 = nn.Linear(self.hidden, 32)
        self.fc5 = nn.Linear(32, self.img_size * self.img_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # h = self.relu(self.fc1(x))
        # return self.fc21(h), self.fc22(h)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        # h = self.relu(self.fc3(z))
        # return self.sigmoid(self.fc4(h))
        h = self.relu(self.fc3(z))
        h = self.relu(self.fc4(h))
        return self.sigmoid(self.fc5(h))

    def forward(self, x):
        x = x.view(-1, self.img_size * self.img_size)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(
            recon_x,
            x.view(-1, self.img_size * self.img_size),
            size_average=False)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

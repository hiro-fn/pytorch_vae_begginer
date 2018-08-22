import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

from net import Net

image_size = 82
batch_size = 16


def make_transform():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    return tf.Compose([
        tf.Resize((image_size, image_size)),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])


def set_dataset(transform):
    trainset = torchvision.datasets.ImageFolder(
        root='data/train', transform=transform)

    testset = torchvision.datasets.ImageFolder(
        root='data/val', transform=transform)

    return trainset, testset


def load_dataset(dataset, shuffle: bool, batch_size: int, num_workers=4):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)


def get_optimizer(net):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return optimizer


def run_train(net, epoch, optimizer, train_loader):
    net.train()
    loss = 0.0

    for (data, _) in train_loader.dataset:
        data = data.cuda()
        inputs = Variable(data)
        optimizer.zero_grad()

        recon_batch, mu, logvar = net(data)

        loss = net.loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        optimizer.step()

    return loss


def run_test(net, epoch, test_loader):
    net.eval()
    test_loss = 0

    for i, (data, _) in enumerate(test_loader.dataset):
        data = data.cuda()
        inputs = Variable(data, volatile=True)

        recon_batch, mu, logvar = net(data)
        loss = net.loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.data[0]

        n = 2
        comparison = torch.cat([
            data[:n],
            recon_batch.view(3, image_size, image_size)[:n]
        ])
        save_image(comparison.cpu(), f'output/result_{epoch}_{i}.png')

    test_loss /= len(test_loader.dataset)

    return test_loss


def main():
    transform = make_transform()
    train_set, test_set = set_dataset(transform)

    epoch = 200
    train_loader = load_dataset(train_set, batch_size=batch_size, shuffle=True)
    test_loader = load_dataset(test_set, batch_size=batch_size, shuffle=False)

    net = Net()
    net.cuda()
    optimizer = get_optimizer(net)

    loss = 0.0
    for epoch in tqdm(range(epoch)):
        loss = run_train(net, epoch, optimizer, train_loader)
        run_test(net, epoch, test_loader)

        loss += loss.item()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, loss / len(train_loader.dataset)))


main() if __name__ == '__main__' else None
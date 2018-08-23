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


def make_transform(is_train):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)


    if is_train:
        image_process = [
            tf.RandomHorizontalFlip(),
            tf.RandomVerticalFlip(),
            tf.RandomGrayscale(0.3),
            tf.RandomCrop(82),
            tf.Resize((image_size, image_size)),
        ]
    else:
        image_process = [
            tf.Resize((image_size, image_size)),
        ]


    to_tensor_process = [
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ]

    process = []
    process.extend(image_process)
    process.extend(to_tensor_process)

    return tf.Compose(process)


def set_dataset(transform, root):
    dataset = torchvision.datasets.ImageFolder(
        root=root, transform=transform)

    return dataset


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
    train_transform = make_transform(True)
    test_transform = make_transform(False)
    train_set = set_dataset(train_transform, 'data/train')
    test_set = set_dataset(test_transform, 'data/val')

    epoch = 1000
    train_loader = load_dataset(train_set, batch_size=batch_size, shuffle=True)
    test_loader = load_dataset(test_set, batch_size=batch_size, shuffle=False)

    net = Net()
    net.cuda()
    optimizer = get_optimizer(net)

    loss = 0.0
    for epoch in tqdm(range(epoch)):
        loss = run_train(net, epoch, optimizer, train_loader)

        if epoch % 10 == 0:
            run_test(net, epoch, test_loader)

        loss += loss.item()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, loss / len(train_loader.dataset)))


main() if __name__ == '__main__' else None
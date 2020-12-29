from __future__ import print_function

from statistics import mean

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torchvision import datasets, transforms


# Test 0_0_1 can mutation be done # yes # how heavy is this task
# can you change the network on the go with torch
# TODO: implment  https://en.wikipedia.org/wiki/Hebbian_theory and neural plasticity
class test_0_0_1_mutation(nn.Module):
    def __init__(self):
        super(test_0_0_1_mutation, self).__init__()

        self.mutation_options = {
            "activation": F.relu,
            "": lambda a, b: nn.Linear(a, b)
        }

        self.mutation_state = nn.ModuleList()
        self.mutation = nn.Sequential(*self.mutation_state)

        self.fc_in = nn.Linear(28, 128)
        self.fc_out = nn.Linear(28 * 128, 10)

    def mutate_state(self):
        self.mutation_state.append(
            nn.Linear(128, 128)
        )
        self.mutation_state.append(nn.ReLU())

    def forward(self, x):
        out = self.fc_in(x)

        out = self.mutation(out)

        out = torch.flatten(out, 1)
        out = self.fc_out(out)
        out = F.log_softmax(out, dim=1)
        return out


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 32}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = td.DataLoader(dataset1, **train_kwargs)
    test_loader = td.DataLoader(dataset2, **test_kwargs)

    model = test_0_0_1_mutation()
    criterion = nn.CrossEntropyLoss()

    model.mutate_state()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    if use_cuda:
        model.cuda()
    print(model)
    losses = []
    for epoch in range(5):
        losses_epoch = []
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_cuda:
                images, targets = images.cuda(), targets.cuda()

            probs = model(images)
            loss = criterion(probs, targets)

            loss.backward()
            optimizer.step()
            losses_epoch.append(loss.item())

            if epoch == 0:
                if i % 200 == 0:
                    model.mutate_state()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


        losses.append(mean(losses_epoch))

    print(model)
    plt.plot(losses)
    plt.show()
    print(model)

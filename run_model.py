import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn
import matplotlib.pyplot as plt
import torchensemble

import Model


def adjust_learning_rate(optimizer, losses):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if losses[0] < losses[1] and losses[1] < losses[2]:
        for param_group in optimizer.param_groups:
            param_group['lr']  *= .5
    return


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
    return loss.item()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output.log(), target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


if __name__ == "__main__":
    optimize_learning = True
    ensemble_learning = True
    batch_size = 128
    num_epochs = 40
    device = torch.device('cuda:1')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    if ensemble_learning:
        model = torchensemble.VotingClassifier(
            estimator = Model.Net().to(device),
            n_estimators = 7,
            cuda=True
        )
        criterion = torch.nn.CrossEntropyLoss()
        model.set_criterion(criterion)
        model.set_optimizer('SGD',lr=.01,momentum=0.5)
        model.fit(train_loader=train_loader, epochs=num_epochs, test_loader=test_loader)
        accuracy = model.evaluate(test_loader)
        print("Final accuracy:", accuracy)
    else:
        model = Model.Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        train_loss = []
        test_accuracy = []
        for epoch in range(1, num_epochs + 1):
            if epoch > 3 and optimize_learning:
                adjust_learning_rate(optimizer, train_loss[-4:-1])
            train_loss.append(train(model, device, train_loader, optimizer, epoch))
            test_accuracy.append(test(model, device, test_loader))
        plt.plot([x for x in range(1, num_epochs+1)], train_loss, "r")
        plt.title("Loss per epoch")
        plt.show()
        plt.plot([x for x in range(1, num_epochs+1)], test_accuracy, "b")
        plt.title("Accuracy per epoch")
        plt.show()

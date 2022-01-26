from NN_architecture import Net
import torch.nn as nn
from Train_Network import train_net


def test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=5):
    model = Net()
    criterion = nn.CrossEntropyLoss()
    loss_train, test_accuracy = train_net(model, train_loader, criterion, optimizer, test_loader, batch_size,
                                          n_epochs=n_epochs)
    return loss_train, test_accuracy

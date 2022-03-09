import numpy as np
from Test_Trained_Network import test_net
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
from collections import deque

def train(epochs, network, optimizer, train_loader,log_interval = 1, writer=None, test_loader=None):
  train_losses=[]
  train_counter=[]
  i=0
  criterion = torch.nn.NLLLoss().to('cuda')
  loss_train_l = deque()
  loss_test_l = deque()

  for epoch in range(epochs):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data= data.to('cuda')
        target = target.to('cuda')
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     train_losses.append(loss.item())
        writer.add_scalar('loss/train', loss, i)
        i+=1
        loss_train_l.append(loss.item())
            # train_counter.append(
            #     (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        # torch.save(network.state_dict(), '/results/model.pth')
        # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
    with torch.no_grad():
        loss_test = 0
        for i, (images, labels) in enumerate(test_loader):
            network.eval()
            outputs = network(images.to('cuda'))
            loss_test += criterion(outputs, labels.to('cuda'))
        loss_test_l.append(loss_test.item())
  return loss_train_l, loss_test_l

def train_net(model, train_loader, criterion, optimizer, test_loader, batch_size, n_epochs=300):
    """train the network by the model
        n_epochs - number of times the NN see all the train data"""
    model.train()  # prep model for training
    loss_train = np.zeros(n_epochs)
    test_accuracy = np.zeros(n_epochs)
    loss_train =[]
    for epoch in range(n_epochs):
        cur_train_loss = 0.0

        # train the model
        for x_data, y_label in train_loader:
            optimizer.zero_grad()  # clear the gradients of all optimized variables
            output = model(x_data)  # forward pass: compute predictions
            # loss = criterion(output, y_label)  # calculate the loss
            loss = F.nll_loss(output, y_label)
            loss.backward()  # backward pass: compute gradient of the loss
            optimizer.step()  # parameter update - perform a single optimization step
            cur_train_loss += loss.item() * x_data.size(0)  # update running training loss
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, loss.item()))
            loss_train.append(loss.item())

            # calculate average loss over an epoch
        train_loss = cur_train_loss / len(train_loader.dataset)



        # test accuracy for this epoch
        test_accuracy[epoch] = test_net(model, test_loader, criterion, batch_size)

    return loss_train, test_accuracy

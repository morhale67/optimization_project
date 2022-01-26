import numpy as np
from Test_Trained_Network import test_net


def train_net(model, train_loader, criterion, optimizer, test_loader, batch_size, n_epochs=30):
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
            loss = criterion(output, y_label)  # calculate the loss
            loss.backward()  # backward pass: compute gradient of the loss
            optimizer.step()  # parameter update - perform a single optimization step
            cur_train_loss += loss.item() * x_data.size(0)  # update running training loss
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, cur_train_loss))
            loss_train.append(loss.item())

            # calculate average loss over an epoch
        train_loss = cur_train_loss / len(train_loader.dataset)



        # test accuracy for this epoch
        test_accuracy[epoch] = test_net(model, test_loader, criterion, batch_size)

    return loss_train, test_accuracy

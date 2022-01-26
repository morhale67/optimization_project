import torch
import numpy as np


def test_net(model, test_loader, criterion, batch_size):

    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()  # prep model for training

    for x_data, y_label in test_loader:
        output = model(x_data)  # forward pass
        loss = criterion(output, y_label)  # calculate the loss
        test_loss += loss.item() * x_data.size(0)  # update test loss
        _, pred = torch.max(output, 1)  # convert output probabilities to predicted class
        correct = np.squeeze(pred.eq(y_label.data.view_as(pred)))  # compare predictions to true label

        # test accuracy
        for i in range(batch_size):
            label = y_label.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # avg test loss
    # test_loss = test_loss/len(test_loader.dataset)
    # print('Test Loss: {:.6f}\n'.format(test_loss))
    #
    # for i in range(10):
    #     if class_total[i] > 0:
    #         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
    #             str(i), 100 * class_correct[i] / class_total[i],
    #             np.sum(class_correct[i]), np.sum(class_total[i])))
    #     else:
    #         print('Test Accuracy of %5s: N/A (no training examples)' % i)

    test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    # print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (test_accuracy, np.sum(class_correct), np.sum(class_total)))
    return test_accuracy

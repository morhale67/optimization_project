from Test_Algorithm import test_optimizer_by_epoch
from Load_Data import load_data_original
import torchvision.transforms as transforms
from torchvision import datasets
import torch
from NN_architecture import Net, init_weights
import matplotlib.pyplot as plt
from datetime import datetime
from Train_Network import train
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
now = datetime.now()

epochs = 100
batch_size = 128
num_workers=12
# loaders
transform = transforms.ToTensor()  # convert data to torch.FloatTensor

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
num_classes= len(train_data.classes)
channels = 1 if len(train_data.data.shape)==3 else train_data.data.shape[-1]
im_size = train_data.data.shape[-2]
# slice the data
# train_data = Subset(train_data, np.arange(num_train_samples))
# test_data = Subset(test_data, np.arange(num_train_samples))

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, num_workers=num_workers)

learning_rate = 1e-3
for learning_rate in [1e-4, 1e-2]:
    for batch_size in [512]:
        #CIFAR 10
        classes = train_data.classes
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('ADAM', now.strftime('%Y%m%d_%H%M%S'), 
                    'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        model = Net(channels=channels, im_size=im_size, num_classes=num_classes).to('cuda')
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        Adam_loss_train = train(epochs, model, optimizer, train_loader, writer=writer)

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                # images, labels = data
                outputs = model(images.to('cuda'))
                writer.add_scalar('loss/test', F.nll_loss(outputs, labels.to('cuda')), i)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions.cpu()):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        # Adam_loss_train, Adam_test_accuracy = test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=n_epochs)


        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('SGD', now.strftime('%Y%m%d_%H%M%S'), 
                    'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        # model = Net(channels=3, im_size=32, num_classes=num_classes).to('cuda')
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # SGD_loss_train, SGD_test_accuracy = test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=n_epochs)
        SGD_loss_train = train(epochs, model, optimizer, train_loader, writer=writer)
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                # images, labels = data
                outputs = model(images.to('cuda'))
                writer.add_scalar('loss/test', F.nll_loss(outputs, labels.to('cuda')), i)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions.cpu()):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('RMSprop', now.strftime('%Y%m%d_%H%M%S'), 
                        'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        # model = Net(channels=3, im_size=32, num_classes=num_classes).to('cuda')
        model.apply(init_weights)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        # RMS_loss_train, RMS_test_accuracy = test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=n_epochs)
        RMS_loss_train = train(epochs, model, optimizer, train_loader, writer=writer)
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                # images, labels = data
                outputs = model(images.to('cuda'))
                writer.add_scalar('loss/test', F.nll_loss(outputs, labels.to('cuda')), i)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions.cpu()):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('ADAdelta', now.strftime('%Y%m%d_%H%M%S'),
                'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        # model = Net(channels=3, im_size=32, num_classes=num_classes).to('cuda')
        model.apply(init_weights)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
        # RMS_loss_train, RMS_test_accuracy = test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=n_epochs)
        RMS_loss_train = train(epochs, model, optimizer, train_loader, writer=writer)
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                # images, labels = data
                outputs = model(images.to('cuda'))
                writer.add_scalar('loss/test', F.nll_loss(outputs, labels.to('cuda')), i)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions.cpu()):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
# len_list = len(Adam_loss_train)
# plt.scatter(range(len_list), Adam_loss_train)
# plt.scatter(range(len_list), SGD_loss_train)
# plt.scatter(range(len_list), RMS_loss_train)
# fig.show()

# # len_list = len(Adam_test_accuracy)
# # plt.scatter(range(len_list), Adam_test_accuracy)
# # plt.scatter(range(len_list), SGD_test_accuracy)
# # plt.scatter(range(len_list), RMS_test_accuracy)
# # plt.show()

# print('end')



# n_epochs = 5
# model = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # optimizer-SGD with alpha=0.01
# loss_train, test_accuracy = train_net(model, train_loader, criterion, optimizer, test_loader, batch_size, n_epochs=n_epochs)
# # test_accuracy = test_net(model, test_loader, criterion, batch_size)
# print('end')






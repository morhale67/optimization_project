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
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.font_manager as font_manager
import matplotlib
from visualize import style_image
now = datetime.now()

epochs = 50
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
font = font_manager.FontProperties(weight='bold',
                                   style='normal', size=16)
matplotlib.rcParams.update({'font.size':22})
learning_rate = 1e-3
for learning_rate in [1e-3]:
    plt.figure()
    for batch_size in [512]:
        #CIFAR 10
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
        classes = train_data.classes
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('ADAM', now.strftime('%Y%m%d_%H%M%S'), 
                    'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        model = Net(channels=channels, im_size=im_size, num_classes=num_classes).to('cuda')
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        adam_loss_train, adam_loss_test = train(epochs, model, optimizer, train_loader, writer=writer, test_loader=test_loader)

        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('SGD', now.strftime('%Y%m%d_%H%M%S'), 
                    'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        sgd_loss_train, sgd_loss_test = train(epochs, model, optimizer, train_loader, writer=writer, test_loader=test_loader)

        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('RMSprop', now.strftime('%Y%m%d_%H%M%S'), 
                        'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        model.apply(init_weights)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        rms_loss_train, rms_loss_test = train(epochs, model, optimizer, train_loader, writer=writer, test_loader=test_loader)

        log_name = '{}_{}_{}={}_{}={}_{}={}'.format('ADAdelta', now.strftime('%Y%m%d_%H%M%S'),
                'lr', str(learning_rate), 'batch_size', str(batch_size), 'epochs', str(epochs))
        writer = SummaryWriter('logs/{}'.format(log_name))
        model.apply(init_weights)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
        ada_loss_train, ada_loss_test = train(epochs, model, optimizer, train_loader, writer=writer, test_loader=test_loader)
    train_plt=plt.figure(figsize=(20, 7))
    plt.plot(np.asarray(adam_loss_train))
    plt.plot(np.asarray(sgd_loss_train))
    plt.plot(np.asarray(rms_loss_train))
    plt.plot(np.asarray(ada_loss_train))
    plt.legend(['ADAM', 'SGD', 'RMSprop', 'ADAdelta'], prop=font)
    plt.xlabel('iteration')
    plt.ylabel('error')
    # plt.title('Optimizer train comparison with learning rate='+str(learning_rate))
    train_plt.savefig(log_name + '_train.png')
    train_plt=plt.figure(figsize=(20, 7))
    plt.plot(np.asarray(adam_loss_test))
    plt.plot(np.asarray(sgd_loss_test))
    plt.plot(np.asarray(rms_loss_test))
    plt.plot(np.asarray(ada_loss_test))
    plt.legend(['ADAM', 'SGD', 'RMSprop', 'ADAdelta'], prop=font)
    plt.xlabel('iteration')
    plt.ylabel('error')
    # plt.title('Optimizer test comparison with learning rate='+str(learning_rate))
    train_plt.savefig(log_name + '_test.png')

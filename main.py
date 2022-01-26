from Test_Algorithm import test_optimizer_by_epoch
from Load_Data import load_data_original
import torch
from NN_architecture import Net
import matplotlib.pyplot as plt

n_epochs = 10

train_loader, test_loader, batch_size = load_data_original()
model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=1)
Adam_loss_train, Adam_test_accuracy = test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=n_epochs)

optimizer = torch.optim.SGD(model.parameters(), lr=1)
SGD_loss_train, SGD_test_accuracy = test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=n_epochs)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1)
RMS_loss_train, RMS_test_accuracy = test_optimizer_by_epoch(optimizer, train_loader, test_loader, batch_size, n_epochs=n_epochs)

len_list = len(Adam_loss_train)
plt.scatter(range(len_list), Adam_loss_train)
plt.scatter(range(len_list), SGD_loss_train)
plt.scatter(range(len_list), RMS_loss_train)
plt.show()

len_list = len(Adam_test_accuracy)
plt.scatter(range(len_list), Adam_test_accuracy)
plt.scatter(range(len_list), SGD_test_accuracy)
plt.scatter(range(len_list), RMS_test_accuracy)
plt.show()

print('end')



# n_epochs = 5
# model = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # optimizer-SGD with alpha=0.01
# loss_train, test_accuracy = train_net(model, train_loader, criterion, optimizer, test_loader, batch_size, n_epochs=n_epochs)
# # test_accuracy = test_net(model, test_loader, criterion, batch_size)
# print('end')






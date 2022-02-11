import torch.nn as nn
import torch.nn.functional as F
import torch

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         hidden_1, hidden_2 = 512, 512  # number of hidden nodes in each layer (512)
#         self.fc1 = nn.Linear(28 * 28, hidden_1)  # linear layer (784 -> hidden_1)
#         self.fc2 = nn.Linear(hidden_1, hidden_2)  # linear layer (n_hidden -> hidden_2)
#         self.fc3 = nn.Linear(hidden_2, 10)  # linear layer (n_hidden -> 10)
#         self.dropout = nn.Dropout(0.2)  # dropout prevents overfitting of data

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)  # flatten image input
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self, channels=3, im_size=28, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 128, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1_drop = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.convs=[]
        self.drops=[]
        # for i in range(6):
        #     self.convs.append(nn.Conv2d(10, 10, kernel_size=5, stride=1, padding=2))
        #     self.drops.append(nn.Dropout2d(0.7))
        self.convs = nn.ModuleList(self.convs)
        self.drops = nn.ModuleList(self.drops)
        self.FC_size = int(64*(im_size/4-3)**2)
        self.fc1 = nn.Linear(self.FC_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1_drop(self.conv1(x)))
        x = F.max_pool2d(self.bn1(x), 2)
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.max_pool2d(self.bn2(x), 2)
        x = x.view(-1, self.FC_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
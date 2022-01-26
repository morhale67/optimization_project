import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1, hidden_2 = 512, 512  # number of hidden nodes in each layer (512)
        self.fc1 = nn.Linear(28 * 28, hidden_1)  # linear layer (784 -> hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)  # linear layer (n_hidden -> hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)  # linear layer (n_hidden -> 10)
        self.dropout = nn.Dropout(0.2)  # dropout prevents overfitting of data

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten image input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


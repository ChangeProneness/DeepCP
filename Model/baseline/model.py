import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, len=0):
        self.len = len
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, 3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(5, 5, 3)
        self.fc1 = nn.Linear(5 * ((self.len - 2)//2 -2), 2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 5 * ((self.len - 2)//2 -2))
        x = self.fc1(x)
        return x

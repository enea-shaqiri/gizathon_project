import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# Define the neural network architecture
class BasicFC(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super(BasicFC, self).__init__()
        self.fc_1 = nn.Linear(input_size, hidden_size_1).double()
        self.fc_2 = nn.Linear(hidden_size_1, hidden_size_2).double()
        self.fc_3 = nn.Linear(hidden_size_2, hidden_size_3).double()
        self.fc_4 = nn.Linear(hidden_size_3, 1).double()
        self.dropout = nn.Dropout(p=0.8)
        self.init_weights()

    def forward(self, x):
        output = self.dropout(torch.relu(self.fc_1(x)))
        output = self.dropout(torch.relu(self.fc_2(output)))
        output = self.dropout(torch.relu(self.fc_3(output)))
        output = self.fc_4(output)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

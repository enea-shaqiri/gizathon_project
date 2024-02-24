import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# Define the neural network architecture
class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.init_weights()

    def forward(self, x):
        # hidden state
        h_0 = Variable(torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size)).to(dtype=x.dtype, device=x.device)
        # internal state
        c_0 = Variable(torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size)).to(dtype=x.dtype, device=x.device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        output = output[:, -1]
        output = torch.relu(output)
        output = self.fc(output)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 30)



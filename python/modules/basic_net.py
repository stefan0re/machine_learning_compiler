import torch.nn as nn
import torch.nn.functional as F


# Define the model
class BasicNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(BasicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size, bias=True)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size, bias=True)
        self.fc3 = nn.Linear(hidden2_size, output_size, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # no activation here because CrossEntropyLoss expects raw logits
        return self.fc3(x)

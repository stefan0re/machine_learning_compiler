#
# Einsum notation with calculation example
# for the BasicNet with our configuration
#

import torch
import torch.nn.functional as F

# model parameters
a = 1
b = 4
c = 64
d = 16
e = 3

# example input
x = torch.randn(a, b)

# weights and biases
W1 = torch.randn(c, b)
bias1 = torch.randn(c)

W2 = torch.randn(d, c)
bias2 = torch.randn(d)

W3 = torch.randn(e, d)
bias3 = torch.randn(e)

# layer 1
z1 = torch.einsum("ab,cb->ac", x, W1) + bias1
z1_relu = F.relu(z1)

# layer 2
z2 = torch.einsum("ac,dc->ad", z1_relu, W2) + bias2
z2_relu = F.relu(z2)

# output layer
output = torch.einsum("ad,ed->ae", z2_relu, W3) + bias3

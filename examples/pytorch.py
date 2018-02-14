import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import softmax

# ------------------------------------------------------- #
#            Logistic Regression from scratch             #
# ------------------------------------------------------- #

W = Variable(torch.zeros(784, 10))
b = Variable(torch.zeros(1, 10))

def pred(x):
    return softmax(torch.matmul(x, W) + b)

def cost(x, y):
    return (pred(x).log() * y).sum(1).mean()

# See an example prediction
pred(Variable(torch.rand(1,784), requires_grad = False))

# ------------------------------------------------------- #
#                  Custom Layer: Dense                    #
# ------------------------------------------------------- #

class Dense(nn.Module):
    def __init__(self, input, out, act = torch.nn.functional.sigmoid):
        super(Dense, self).__init__()
        self.act = act
        self.W = Parameter(torch.randn(input, out))
        self.b = Parameter(torch.randn(1, out))

    def forward(self, x):
        return self.act(torch.matmul(x, self.W) + self.b)

d = Dense(10, 5, torch.nn.functional.relu)
x = Variable(torch.rand(1, 10), requires_grad = False)
d(x)

# ------------------------------------------------------- #
#                  RNN from scratch                       #
# ------------------------------------------------------- #

class RNN(nn.Module):
    def __init__(self, input, out):
        super(RNN, self).__init__()
        self.Wi = Parameter(torch.randn(input, out))
        self.Wh = Parameter(torch.randn(out, out))
        self.b = Parameter(torch.randn(1, out))

    def forward(self, h, x):
        Wi, Wh, b = self.Wi, self.Wh, self.b
        h = (torch.matmul(x, Wi) + torch.matmul(h, Wh) + b).tanh()
        return (h, h)

rnn = RNN(10, 5)
h = Variable(torch.rand(1, 5), requires_grad = False)
xs = [Variable(torch.rand(1, 10), requires_grad = False) for _ in range(10)]
ys = []

for x in xs:
    (h, y) = rnn(h, x)
    ys.append(y)

# Output hidden state and sequence
h, ys

# ------------------------------------------------------- #
#                  Recursive Net                          #
# ------------------------------------------------------- #

# TODO: similar to Julia

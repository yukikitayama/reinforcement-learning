from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
from datetime import datetime


# Parameter
LOG_DIR = '../tensorboard/log_' + datetime.now().strftime('%Y%m%d')


def check_tensor():
    x = torch.rand(5, 3)
    print('rand(5, 3)', x)
    print('torch.cuda.is_available()', torch.cuda.is_available())

    a = torch.FloatTensor(3, 2)
    print('FloatTensor(3, 2)', a)
    print('zero_()', a.zero_())
    print('After zero_()', a)

    n = np.zeros((3, 2))
    print('np.zeros', n)
    b = torch.tensor(n)
    print('torch.tensor', b)

    n = np.zeros(shape=(3, 2), dtype=np.float32)
    print('np.zeros to torch.tensor', torch.tensor(n))

    a = torch.tensor([1, 2, 3])
    print('Before sum', a)
    s = a.sum()
    print('After sum', s)
    print('s.item()', s.item())
    print(torch.tensor(1))

    # a = torch.FloatTensor([2, 3])
    # print(a)
    # ca = a.cuda()
    # print(ca)


def check_gradient():
    # Gradient
    v1 = torch.tensor([1.0, 1.0], requires_grad=True)
    v2 = torch.tensor([2.0, 2.0])
    v_sum = v1 + v2
    v_res = (v_sum*2).sum()
    print(v_res)
    print(v1.is_leaf, v2.is_leaf)
    print(v_sum.is_leaf, v_res.is_leaf)
    print(v1.requires_grad)
    print(v2.requires_grad)
    print(v_sum.requires_grad)
    print(v_res.requires_grad)
    v_res.backward()
    print(v1.grad)
    print(v2.grad)


def check_neural_network():
    s = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Dropout(p=0.3),
        nn.Softmax(dim=1)
    )
    print('Print neural network sequential')
    print(s)
    print('Output of nn.Sequential(tensor input)')
    print(s(torch.FloatTensor([[1, 2]])))


class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == '__main__':
    # Tensor
    check_tensor()

    # Gradient
    check_gradient()

    # Neural network
    check_neural_network()

    # Module subclassing
    net = OurModule(num_inputs=2, num_classes=3)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(v.shape)
    print(net)
    print(out)

    # TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR)
    funcs = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan}
    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)
    writer.close()

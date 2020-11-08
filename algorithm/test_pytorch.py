"""
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
from datetime import datetime


# Parameter
LOG_DIR = '../tensorboard/log_' + datetime.now().strftime('%Y%m%d')


def numpy_nn():
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)
    learning_rate = 1e-6

    for t in range(500):
        # Forward pass
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Loss function
        loss = np.square(y_pred - y).sum()
        print(t, loss)

        # Backpropagation
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def torch_manual_nn():
    dtype = torch.float
    device = torch.device('cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10
    learning_rate = 1e-6

    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    for t in range(500):
        # Forward pass
        h = x.mm(w1)  # torch.mm is matrix multiplication
        h_relu = h.clamp(min=0)  # torch.clamp sets y = min if x < min
        y_pred = h_relu.mm(w2)

        # Loss function
        # .sum() makes tensor, but .item() makes it a scalar
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)
            print((y_pred - y).pow(2).sum())

        # Backpropagation
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()  # torch.clone makes a copy of input
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def torch_autograd():
    dtype = torch.float
    device = torch.device('cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10
    learning_rate = 1e-6

    x = torch.randn(N, D_in, device=device, dtype=dtype, requires_grad=False)
    y = torch.randn(N, D_out, device=device, dtype=dtype, requires_grad=False)
    # setting requires_grad=True says we want to compute gradients with respect
    # to weight tensors during backpropagation
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    for t in range(500):
        # Forward pass
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        # The above is 2 layer nn with relu
        # we do not make intermediate results because we do not implement back-
        # propagation by hand

        # Loss function
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print('t', t)
            print('loss.item()', loss.item())
            print('loss', loss)

        # Automatic differentiation backpropagation
        loss.backward()
        # The above computes the gradient of the loss with respect to all Tensors
        # with requires_grad=True. After this call, w1.grad and w2.grad will be
        # tensors holding the gradient of the loss with respect to w1 and w2.

        if t % 100 == 99:
            print('w1.grad', w1.grad)

        # Manual gradient descent, which is enabled by torch.no_grad()
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            # Manually zero the gradients after the gradient descent
            w1.grad.zero_()
            w2.grad.zero_()


def nn_manual_update():
    N, D_in, H, D_out = 64, 1000, 100, 10
    learning_rate = 1e-4

    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Linear Module computes output from input using linear function, and holds
    # internal Tensors for its weight and bias.
    model = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out),
    )

    # Loss function
    loss_fn = nn.MSELoss(reduction='sum')

    for t in range(500):

        # Forward pass
        # pass a Tensor input to Module and produce a Tensor output
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print('t', t)
            print('loss.item()', loss.item())
            print('loss', loss)

        # Zero the gradients before backpropagation
        model.zero_grad()

        if t % 100 == 99:
            print('after model.zero_grad()')
            print(list(model.parameters())[0].grad)

        # Backpropagation
        # Compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, each Module parameters are stored
        # in Tensors with requires_grad=True
        loss.backward()

        if t % 100 == 99:
            print('after loss.backward()')
            print(list(model.parameters())[0].grad)

        # Manual gradient descent. That is why torch.no_grad()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad


def torch_optimizer():
    N, D_in, H, D_out = 64, 1000, 100, 10
    learning_rate = 1e-4

    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out),
    )

    loss_fn = nn.MSELoss(reduction='sum')

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for t in range(500):

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print('t', t)
            print('loss.item()', loss.item())

        # Before backpropagation, use optimizer object to zero all the gradients
        # of the learnable weights of the model, because by default gradients are
        # accumulated in buffer.
        optimizer.zero_grad()

        loss.backward()

        # Optimizer step function updates the parameters
        optimizer.step()


def custom_nn_module():
    class Net(nn.Module):
        def __init__(self, D_in, H, D_out):
            super(Net, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Linear(H, D_out),
            )

        def forward(self, x):
            return self.model(x)

    N, D_in, H, D_out = 64, 1000, 100, 10
    learning_rate = 1e-4

    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = Net(D_in, H, D_out)

    loss_fn = nn.MSELoss(reduction='sum')

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for t in range(500):

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print('t', t)
            print('loss.item()', loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
    # Numpy neural network
    # numpy_nn()

    # Torch manual neural network
    # torch_manual_nn()

    # Automatic differentiation
    # torch_autograd()

    # nn Module manual update
    # nn_manual_update()

    # Optimizer
    # torch_optimizer()

    # Custom nn Module
    custom_nn_module()

    # Tensor
    # check_tensor()

    # Gradient
    # check_gradient()

    # Neural network
    # check_neural_network()

    # Module subclassing
    # net = OurModule(num_inputs=2, num_classes=3)
    # v = torch.FloatTensor([[2, 3]])
    # out = net(v)
    # print(v.shape)
    # print(net)
    # print(out)

    # TensorBoard
    # writer = SummaryWriter(log_dir=LOG_DIR)
    # funcs = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan}
    # for angle in range(-360, 360):
    #     angle_rad = angle * math.pi / 180
    #     for name, fun in funcs.items():
    #         val = fun(angle_rad)
    #         writer.add_scalar(name, val, angle)
    # writer.close()

import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init

import math
import numpy as np
import sys

class Conv2dPatternFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation,
                                                      groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class _ConvNdPattern(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNdPattern, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNdPattern, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv2dPattern(_ConvNdPattern):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', mask=torch.zeros(1), check_grad=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.kernel_size = kernel_size
        self.mask = mask

        super(Conv2dPattern, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    # @weak_script_method
    def forward(self, input):

        if (self.mask.shape != self.kernel_size):
            print("[Warning]: Mask size is not consistent",self.mask.shape,self.kernel_size)
            self.mask = torch.ones(self.kernel_size)

        return Conv2dPatternFunction.apply(input, self.weight * self.mask, self.bias, self.stride,
                                             self.padding, self.dilation, self.groups)



if __name__== "__main__":

    # x = torch.autograd.Variable(torch.randn(1, 3, 10, 10), requires_grad=True)

    # BL.weight = TC.weight
    # BL.bias = TC.bias
    #
    # y = TC(x)
    # b = BL(x)


    # print(y.shape)


    # BL.zero_grad()
    # TC(x).backward(torch.randn(1, 5,8,8))
    # print(x.grad.data)

    # print(TC.weight-BL.weight)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = Conv2d(3,1,3)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1,9)
            return x

    class Net_Pattern(nn.Module):
        def __init__(self):
            super(Net_Pattern, self).__init__()

            mask = torch.tensor([[0,1,1],[1,1,1],[1,0,0]])

            self.conv = Conv2dPattern(3,1,3, mask=mask)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1,9)
            return x

    def trainer(model, optimizer,x,y):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model_pattern = Net().to(device)
    model_pattern.conv.weight = model.conv.weight
    model_pattern.conv.bias = model.conv.bias
    optimizer_pattern = torch.optim.Adam(model_pattern.parameters(), lr=0.01)


    # B, C, H, W = 10, 3, 4, 4
    # x = torch.randn(B, C, H, W)
    # y = torch.where(x > x.view(B, C, -1).mean(2)[:, :, None, None], torch.tensor([1.]), torch.tensor([0.]))
    #
    # print(x.shape)
    # print(x)
    # print(y)
    #
    # print(x.mul(y))
    #
    # sys.exit(0)


    x = torch.randn(1, 3, 5, 5)
    y = torch.empty(1, dtype=torch.long).random_(5)


    # trainer(model, optimizer,x,y)
    # trainer(model_pattern, optimizer_pattern,x,y)


    # x = torch.tensor([[[[1,1,1,1,1],
    #       [1,1,1,1,1],
    #       [1,1,1,1,1],
    #       [1,1,1,1,1],
    #       [1,1,1,1,1]]]],dtype=torch.float32)
    # print(x.shape)

    x = torch.randn(10, 3, 5, 5)
    mask = torch.tensor([[0,1,1],[1,1,1],[1,0,0]],dtype=torch.float32)
    TC = Conv2dPattern(3,1,3, mask=mask)
    print(TC.weight)
    print(TC(x))

    # print(model.conv.weight)
    # print(model_pattern.conv.weight)


    #
    # print(model.conv.weight)
    # print(model_pattern.conv.weight)



    # model.eval()
    # model_pattern.eval()
    # x = torch.randn(1, 3, 5, 5)
    # output = model(x)
    # output_pattern = model_pattern(x)
    #
    # print(output)
    # print(output_pattern)
    #
    # sys.exit(0)

    #
    # print(model.conv.weight)






    #
    # BL = Conv2d(3,5,3)
    # TC = Conv2dPattern(3,5,3)


    # #
    # import sys
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # print(input.shape)
    # print(target.shape)

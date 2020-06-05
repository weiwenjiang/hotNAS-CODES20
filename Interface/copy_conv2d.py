import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import init
from torch.nn.modules.utils import _single, _pair, _triple
from torch._jit_internal import List

#
# def quantize(x, num_int_bits, num_frac_bits, signed=True):
#
#     precision = 1 / 2 ** num_frac_bits
#     # print(precision)
#     if signed:
#         bound = 2 ** (num_int_bits - 1)
#         lower_bound = -1*bound
#         upper_bound = bound - precision
#     else:
#         bound = 2 ** num_int_bits
#         lower_bound = 0
#         upper_bound = bound - precision
#
#     return torch.clamp(x.div(precision).int().float().mul(precision), lower_bound, upper_bound)

def quantize(x, num_int_bits, num_frac_bits, signed=True):
    precision = 1 / 2 ** num_frac_bits
    x = torch.round(x / precision) * precision
    if signed is True:
        bound = 2 ** (num_int_bits - 1)
        return torch.clamp(x, -bound, bound - precision)
    else:
        bound = 2 ** num_int_bits
        return torch.clamp(x, 0, bound - precision)

class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
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
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'



class Conv2d_Custom(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 is_pattern=False, pattern={}, pattern_ones=-1,
                 is_quant=False, quan_paras=[], is_std_conv=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.pattern = pattern
        self.is_pattern = is_pattern
        self.pattern_ones = pattern_ones
        self.is_quant = is_quant
        self.quan_paras = quan_paras
        self.is_std_conv = is_std_conv
        super(Conv2d_Custom, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)



    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return  F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def check_layer(self, print_patter=False):
        print("Pattern:",self.is_pattern)
        if print_patter:
            print("\t\tpatterns:",self.pattern)
        print("\t\tnumber of ones in pattern:", self.pattern_ones)
        print("Quantization:", self.is_quant)
        print("\t\tInt: {}, Fraction: {}, Sign: {}".format(self.quan_paras[0],self.quan_paras[1],self.quan_paras[2]))


    def forward(self, input):
        w = self.weight
        if self.is_std_conv:
            v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
            w = (w - m) / torch.sqrt(v + 1e-10)



        if self.is_pattern and not self.is_quant:
            return self.conv2d_forward(input, w * self.pattern)
        elif not self.is_pattern and self.is_quant:
            return self.conv2d_forward(input, quantize(w, self.quan_paras[0], self.quan_paras[1],
                                                       signed=self.quan_paras[2]))
        elif self.is_pattern and self.is_quant:
            return self.conv2d_forward(input, quantize(w * self.pattern, self.quan_paras[0],
                                                       self.quan_paras[1],signed=self.quan_paras[2]))
        else:
            return self.conv2d_forward(input, w)

        # print("Pattern take effects")

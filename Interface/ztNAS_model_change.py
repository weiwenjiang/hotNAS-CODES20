from torch import nn
import torch
import sys
from utility import is_same
import pattern_kernel

def get_last_attr_idx(model,seq):

    last_not_digit = 0
    pre_attr = model
    last_attr = []


    a = model
    for idx in range(len(seq)):
        var = seq[idx]
        if var.isdigit():
            a = a[int(var)]
        else:
            pre_attr = a
            a = getattr(a, var)
            last_not_digit = idx
            last_attr = a
    return pre_attr,last_attr,last_not_digit


def ztNAS_modify_kernel_shape(model,layer, layer_name,var_k,increase=True):
    [M, N, K, S, G, P, b] = (
        layer.out_channels, layer.in_channels, is_same(layer.kernel_size),
        is_same(layer.stride), layer.groups, is_same(layer.padding), layer.bias)

    ## Weiwen: 03-29
    ## Step 1: Translate layer name to locate layer module
    ##
    seq = layer_name.split(".")
    (pre_attr,last_attr,last_not_digit) = get_last_attr_idx(model, seq)

    ## Weiwen: 03-29
    ## Step 2: Backup weights and bias if exist
    ##
    if b:
        ori_para_w = model.state_dict()[layer_name + ".weight"][:]
        ori_para_b = model.state_dict()[layer_name + ".bias"][:]
    else:
        ori_para_w = model.state_dict()[layer_name + ".weight"][:]

    ## Weiwen: 03-29
    ## Step 3: Translate layer name to locate layer module
    ##
    if last_not_digit == len(seq) - 1:
        # last one is the attribute, directly setattr
        new_conv = nn.Conv2d(N, M, kernel_size=(K + var_k, K + var_k), stride=(S, S), padding=(int(P + var_k/2), int(P + var_k/2)), groups=G, bias=b)
        setattr(pre_attr, seq[-1], new_conv)
    elif last_not_digit == len(seq) - 2:
        # one index last_attr[]
        new_conv = nn.Conv2d(N, M, kernel_size=(K + var_k, K + var_k), stride=(S, S), padding=(int(P + var_k/2), int(P + var_k/2)), groups=G, bias=b)
        last_attr[int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-2], last_attr)
    elif last_not_digit == len(seq) - 3:
        # two index last_attr[][]
        new_conv = nn.Conv2d(N, M, kernel_size=(K + var_k, K + var_k), stride=(S, S), padding=(int(P + var_k/2), int(P + var_k/2)), groups=G, bias=b)
        last_attr[int(seq[-2])][int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-3], last_attr)
    else:
        print("more than 2 depth of index from last layer is not support!")
        sys.exit(0)

    ## Weiwen: 03-29
    ## Step 4: Setup new parameters from backup
    ##
    if b:
        model.state_dict()[layer_name + ".bias"][:] = ori_para_b

    if var_k>0:
        pad_fun = torch.nn.ZeroPad2d(int(var_k/2))
        model.state_dict()[layer_name + ".weight"][:] = pad_fun(ori_para_w)

    return model



def ztNAS_add_kernel_mask(model,layer, layer_name,mask):
    [M, N, K, S, G, P, b] = (
        layer.out_channels, layer.in_channels, is_same(layer.kernel_size),
        is_same(layer.stride), layer.groups, is_same(layer.padding), layer.bias)

    ## Weiwen: 03-29
    ## Step 1: Translate layer name to locate layer module
    ##
    seq = layer_name.split(".")
    (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)

    ## Weiwen: 03-29
    ## Step 2: Backup weights and bias if exist
    ##
    if b:
        ori_para_b = model.state_dict()[layer_name + ".bias"][:]
    ori_para_w = model.state_dict()[layer_name + ".weight"][:]

    ## Weiwen: 03-29
    ## Step 3: Translate layer name to locate layer module
    ##
    if last_not_digit == len(seq) - 1:
        # last one is the attribute, directly setattr
        new_conv = pattern_kernel.Conv2dPattern(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=b, mask=mask)
        setattr(pre_attr, seq[-1], new_conv)
    elif last_not_digit == len(seq) - 2:
        # one index last_attr[]
        new_conv = pattern_kernel.Conv2dPattern(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=b, mask=mask)
        last_attr[int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-2], last_attr)
    elif last_not_digit == len(seq) - 3:
        # two index last_attr[][]
        new_conv = pattern_kernel.Conv2dPattern(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=b, mask=mask)
        last_attr[int(seq[-2])][int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-3], last_attr)
    else:
        print("more than 2 depth of index from last layer is not support!")
        sys.exit(0)

    ## Weiwen: 03-29
    ## Step 4: Setup new parameters from backup
    ##
    if b:
        model.state_dict()[layer_name + ".bias"][:] = ori_para_b
    model.state_dict()[layer_name + ".weight"][:] = ori_para_w


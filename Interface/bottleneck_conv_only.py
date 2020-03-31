from torchvision import models

from torchvision.models import *
from torch import nn
import torch
import sys
import math
sys.path.append("../Performance_Model")
import PM_Config
import PM_Layer
import PM_FPGA_Template
from search_space import *
from CONV_PM_IF import *
import argparse



def get_max_k(model):
    max_k = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            cur_k = is_same(layer.kernel_size)
            if cur_k > max_k:
                max_k = cur_k
    return  max_k

def get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p):
    input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
    cTT = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            input_shape = list(input.shape)
            input_shape[1] = layer.in_channels
            input = torch.Tensor(torch.Size(input_shape)).to(torch.float32)
            input = layer(input)

            [B, M, N, R, C, K, S, T, P] = (
                1, layer.out_channels, layer.in_channels, input.shape[2], input.shape[3], is_same(layer.kernel_size),
                is_same(layer.stride), tell_conv_type(layer.in_channels, layer.groups), is_same(layer.padding))

            if T == "cconv":
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                                                                   HW_constraints["r_BRAM_Size"], HW_constraints["r_BRAM"],
                                                                   HW_constraints["BITWIDTH"])

                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv", P)
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                                        Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size,
                                                        BITWIDTH)
                if acc_1.Success==False:
                    return -1
                else:
                    cTT += acc_1.get_layer_latency(Layer)[0]
                    print(layer_name,acc_1.get_layer_latency(Layer))

        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer,
                                                                                                      nn.AvgPool2d):
            input = layer(input)
    # print("\tTotal Time:", (cTT) / 10 ** 5)
    return cTT / 10 ** 5




if __name__== "__main__":

    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='vgg16_bn'
    )
    parser.add_argument(
        '-c', '--cconv',
        default='70, 36, 256, 32, 3, 12, 10, 8',
        help="hardware desgin of cconv",
    )

    args = parser.parse_args()
    model_name = args.model
    model = globals()[model_name]()

    print(args.cconv)
    print(args.cconv.split(","))
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in args.cconv.split(",")]
    # Tk = get_max_k(model)
    # Tn = math.floor((HW_constraints["r_DSP"]-Tm) / Tm)



    print("="*10,model_name,"performance analysis:")

    total_lat = get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p)

    print(total_lat)
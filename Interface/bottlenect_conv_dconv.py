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
from ztNAS_model_change import *
import copy_conv2d

def get_max_k(model):
    max_k = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            cur_k = is_same(layer.kernel_size)
            if cur_k > max_k:
                max_k = cur_k
    return  max_k

def get_performance(model, HW1, HW2,device=None):
    input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
    cTT = 0
    dTT = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer,copy_conv2d.Conv2d_Custom):
            input_shape = list(input.shape)
            input_shape[1] = layer.in_channels
            input = torch.Tensor(torch.Size(input_shape)).to(torch.float32)
            if device is not None:
                input = input.to(device)
            input = layer(input)




            [B, M, N, R, C, K, S, T, P] = (
                1, layer.out_channels, layer.in_channels, input.shape[2], input.shape[3], is_same(layer.kernel_size),
                is_same(layer.stride), tell_conv_type(layer.in_channels, layer.groups), is_same(layer.padding))

            if T == "cconv":
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = HW2

                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (
                HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                HW_constraints["r_BRAM_Size"], HW_constraints["r_BRAM"],
                HW_constraints["BITWIDTH"])
                print('''quan_paras["{}"] = [0, 16, True]'''.format(layer_name))
                # print("\t",layer_name,M, N, R, C, K, S, T)
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv", P)
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                                        Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size,
                                                        BITWIDTH)
                if acc_1.Success == False:
                    return -1
                else:
                    if isinstance(layer, copy_conv2d.Conv2d_Custom):
                        perf = acc_1.get_layer_latency(Layer, layer.pattern_ones, layer.quan_paras)
                    else:
                        perf = acc_1.get_layer_latency(Layer)
                    cTT += perf[0]
                    # # if perf[1] == "loading IFM":
                    # if perf[1] == "loading Weight":
                    # # if perf[1] == "computing":
                    #     print("cconv",layer_name, "Kernel:", K, perf[0] / 10 ** 5, perf[1], [x / 10 ** 5 for x in perf[2]])

            elif T == "dconv":
                print('''quan_paras["{}"] = [0, 16, True]'''.format(layer_name))
                # print("\t",layer_name,M, N, R, C, K, S, T)
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = HW1
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (
                                            HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                                            HW_constraints["r_BRAM_Size"], HW_constraints["r_BRAM"],
                                            HW_constraints["BITWIDTH"])
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "dconv", P)
                acc_2 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                                        Tk, W_p, I_p, O_p, "dconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size,
                                                        BITWIDTH)
                if acc_2.Success == False:
                    return -1
                else:
                    if isinstance(layer, copy_conv2d.Conv2d_Custom):
                        perf = acc_2.get_layer_latency(Layer, layer.pattern_ones, layer.quan_paras)
                    else:
                        perf = acc_2.get_layer_latency(Layer)


                    dTT+=perf[0]

                    # if perf[1] == "loading Weight":
                    # # if perf[1] == "loading IFM":
                    # # if perf[1] == "computing":
                    #     print("dconv",layer_name, "Kernel:", K, perf[0] / 10 ** 5, perf[1], [x / 10 ** 5 for x in perf[2]])

        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer,
                                                                                                      nn.AvgPool2d):
            input = layer(input)

    return (cTT+dTT) / 10 ** 5




if __name__== "__main__":

    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='mnasnet0_5'
    )
    parser.add_argument(
        '-c', '--cconv',
        default="100, 16, 32, 32, 3, 6, 10, 14",
        help="hardware desgin of cconv",
    )
    parser.add_argument(
        '-dc', '--dconv',
        default="832, 1, 32, 32, 5, 6, 10, 14",
        help="hardware desgin of cconv",
    )

    args = parser.parse_args()
    model_name = args.model

    if "proxyless" in model_name:
        model = torch.hub.load('mit-han-lab/ProxylessNAS', model_name)
    elif "FBNET" in model_name:
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'fbnetc_100')
    else:
        model = globals()[model_name]()

    print(model)

    for name, para in model.named_parameters():
        print(name)

    HW1 = [int(x.strip()) for x in args.dconv.split(",")]
    HW2 = [int(x.strip()) for x in args.cconv.split(",")]

    print("="*10,model_name,"performance analysis:")
    total_lat = get_performance(model, HW1, HW2)
    print(total_lat/2)

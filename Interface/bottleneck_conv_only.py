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
                    perf = acc_1.get_layer_latency(Layer)
                    cTT += perf[0]
                    print(layer_name,perf[0]/10**5,perf[1],[x/10**5 for x in perf[2]])

        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer,
                                                                                                      nn.AvgPool2d):
            input = layer(input)
    # print("\tTotal Time:", (cTT) / 10 ** 5)
    return cTT / 10 ** 5




if __name__== "__main__":

    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='resnet18'
    )
    parser.add_argument(
        '-c', '--cconv',
        default="70, 36, 64, 64, 7, 18, 6, 6",
        help="hardware desgin of cconv",
    )

    args = parser.parse_args()
    model_name = args.model
    model = globals()[model_name]()





    print(model)

    for name,para in model.named_parameters():
        print(name)

    print("="*100)

    print(args.cconv)
    print(args.cconv.split(","))
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in args.cconv.split(",")]
    # Tk = get_max_k(model)
    # Tn = math.floor((HW_constraints["r_DSP"]-Tm) / Tm)


    print(model)



    print("="*10,model_name,"performance analysis:")

    total_lat = get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p)

    print(total_lat)


    conv_modify = {}
    conv_modify["layer4.1.conv1"] = (dict(model.named_modules())["layer4.1.conv1"], 512, 440, ["layer4.1.bn1","layer4.1.conv2"])
    conv_modify["layer4.1.conv2"] = (dict(model.named_modules())["layer4.1.conv2"], 440, 512, [])


    bn_modifiy = {}
    bn_modifiy["layer4.1.bn1"] = (dict(model.named_modules())["layer4.1.bn1"], 440)

    print("="*100)
    ztNAS_cut_channel(model, conv_modify, bn_modifiy)

    print("=" * 10, model_name, "performance analysis:")

    total_lat = get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p)
    print(total_lat)
    sys.exit(0)
    #
    # print("="*100)
    #
    #
    # W = dict(model.named_parameters())["layer4.0.conv1.weight"]
    # Norm2 = W.norm(dim=(2, 3))
    #
    # Filter_OFM_Sum = Norm2.sum(dim=1)
    # Filter_IFM_Sum = Norm2.sum(dim=0)
    #
    # Filter_OFM_Norm2_TopK = (Filter_OFM_Sum.topk(480))
    # Filter_IFM_Norm2_TopK = (Filter_IFM_Sum.topk(210))
    #
    # print(W.shape)
    # print(Norm2.shape)
    # print(W[Filter_OFM_Norm2_TopK[1]].shape)
    #
    # W = W.transpose(0,1)[Filter_IFM_Norm2_TopK[1]].transpose(0,1)
    # print(W.shape)
    #
    #
    # # print(W[Filter_OFM_Norm2_TopK[1]].shape)
    # sys.exit(0)
    #
    #
    #
    # print(Norm2.shape)
    # print(W[Norm2_TopK[1]])
    # # print(tuple(Norm2_TopK[1]))
    # # print( dict(model.named_parameters())["layer4.1.conv1.weight"][Norm2_TopK[1]] )
    # #
    # # a = torch.randint(0, 10, [3, 3, 3, 3])
    # # b = torch.LongTensor([[1,1,1], [2,2,2], [0, 0, 0]]).t()
    # # print(a,a.shape)
    # # print(b,b.shape)
    # # print(a[tuple(b)])
    #
    # # t = torch.tensor([[5.7, 1.4, 9.5], [1.6, 6.1, 4.3], [5.0, 1.1, 9.5], [1.6, 3.1, 4.3], [4.7, 6.4, 9.5], [0.6, 4.1, 4.3]])
    # #
    # # t_norm = t.norm(dim=(1))
    # #
    # # print(t_norm)
    # # values, indices = t_norm.topk(2)
    # #
    # #
    # #
    # # print(values)
    # # print(indices)
    # # print(t[indices])
    # sys.exit(0)
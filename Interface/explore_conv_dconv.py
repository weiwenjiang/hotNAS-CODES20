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
import copy



def get_max_k_d(model):
    max_k,d_max_k = 0,0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if (tell_conv_type(layer.in_channels,layer.groups)=="cconv"):
                cur_k = is_same(layer.kernel_size)
                if cur_k > max_k:
                    max_k = cur_k
            else:
                cur_k = is_same(layer.kernel_size)
                if cur_k > d_max_k:
                    d_max_k = cur_k

    return  max_k,d_max_k

def get_performance(model, HW1, HW2):
    input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
    cTT = 0
    dTT = 0
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
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = HW2
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (
                                            HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                                            HW_constraints["r_BRAM_Size"], HW_constraints["r_BRAM"],
                                            HW_constraints["BITWIDTH"])
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv", P)
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                                        Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size,
                                                        BITWIDTH)
                if acc_1.Success:
                    cTT += acc_1.get_layer_latency(Layer)[0]
                else:
                    return -1
            elif T == "dconv":
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = HW1
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (
                                            HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                                            HW_constraints["r_BRAM_Size"], HW_constraints["r_BRAM"],
                                            HW_constraints["BITWIDTH"])
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "dconv", P)
                acc_2 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                                        Tk, W_p, I_p, O_p, "dconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size,
                                                        BITWIDTH)

                dTT += acc_2.get_layer_latency(Layer)[0]

        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer,
                                                                                                      nn.AvgPool2d):
            input = layer(input)

    return (cTT+dTT) / 10 ** 5



def do_exploration(model):
    (rangeTm,rangeTc,rangeTr,range_Wp,range_Ip,range_Op) = search_space['hw_cd_cconv']
    (d_rangeTm, d_rangeTc, d_rangeTr) = search_space['hw_cd_dconv']

    best_lat = 999999999999
    best_design = []
    Tk,d_Tk = get_max_k_d(model)


    for W_p in range_Wp:
        for I_p in range_Ip:
            for O_p in range_Op:
                if (W_p+I_p+O_p)*HW_constraints["BITWIDTH"] > HW_constraints["r_Ports_BW"]:
                    continue

                for d_Tm in d_rangeTm:
                    d_Tn = 1
                    for Tm in rangeTm:
                        Tn = math.floor((HW_constraints["r_DSP"]-d_Tm) / Tm)
                        if HW_constraints["BITWIDTH"] == 16 and d_Tm + Tm*Tn > HW_constraints["r_DSP"]:
                            continue
                        elif HW_constraints["BITWIDTH"] == 32 and 5*(d_Tm + Tm*Tn) > HW_constraints["r_DSP"]:
                            continue

                        for d_Tc in d_rangeTc:
                            for d_Tr in d_rangeTr:
                                HW1 = [d_Tm, d_Tn, d_Tr, d_Tc, d_Tk, W_p, I_p, O_p]
                                for Tc in rangeTc:
                                    for Tr in rangeTr:
                                        HW2 = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
                                        cur_lat = get_performance(model, HW1,HW2)
                                        print("==",HW1,HW2,cur_lat)
                                        # print("'", Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p, "':", cur_lat,best_lat)
                                        if cur_lat!=-1 and cur_lat<best_lat:
                                            best_lat = cur_lat
                                            best_design = [copy.deepcopy(HW1),copy.deepcopy(HW2)]
    return best_lat,best_design



if __name__== "__main__":

    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='mnasnet0_5'
    )
    args = parser.parse_args()
    model_name = args.model
    model = globals()[model_name]()

    best_lat,best_design = do_exploration(model)

    print(model_name, best_lat,best_design)

from torchvision import models

from torchvision.models import *
from torch import nn
import torch
import sys
import math
sys.path.append("../Performance_Model")
sys.path.append("../")
import cifar10_models
import PM_Config
import PM_Layer
import PM_FPGA_Template
from search_space import *
from CONV_PM_IF import *
import argparse
from ztNAS_model_change import *
import copy_conv2d

from utility import *

def get_max_k(model):
    max_k = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            cur_k = is_same(layer.kernel_size)
            if cur_k > max_k:
                max_k = cur_k
    return  max_k

def get_performance(model, dataset_name, HW1, HW2,device=None):
    if dataset_name == "imagenet":
        input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
    elif dataset_name == "cifar10":
        input = torch.Tensor(torch.Size([1, 3, 32, 32])).to(torch.float32)

    cTT = 0
    dTT = 0

    count = [0,0,0,0]

    cconv_quan_ss = []
    cconv_quan_sn = []
    quan_idx = 0
    cconv_pattern = {}

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

                # w = model.state_dict()[layer_name + ".weight"]
                # x = max(abs(float(w.min())), abs(float(w.max())))
                # int_num, frac_num = re_quantize(x, 16, True)
                # print('''quan_paras["{}"] = [{}, {}, True]'''.format(layer_name, int_num, frac_num))

                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = HW2

                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (
                HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                HW_constraints["r_BRAM"], HW_constraints["r_BRAM_Size"],
                HW_constraints["BITWIDTH"])

                # print("\t",layer_name,M, N, R, C, K, S, T)
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv", P)
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                                        Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size,
                                                        BITWIDTH)
                if acc_1.Success == False:
                    print(Tm, Tn, Tr, Tc,Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)
                    return -1
                else:
                    if isinstance(layer, copy_conv2d.Conv2d_Custom):
                        perf = acc_1.get_layer_latency(Layer, layer.pattern_ones, layer.quan_paras)
                    else:
                        perf = acc_1.get_layer_latency(Layer)

                    # print(perf[0])
                    cTT += perf[0]

                    if perf[1] == "loading Weight":
                        w = model.state_dict()[layer_name + ".weight"]

                        # # For conv_std only
                        # if True:
                        #     v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
                        #     w = (w - m) / torch.sqrt(v + 1e-10)

                        x = max(abs(float(w.min())), abs(float(w.max())))

                        int_num, frac_num = re_quantize(x, 16, True)
                        print('''quan_paras["{}"] = [{}, q_list[{}], True]'''.format(layer_name, int_num, quan_idx))
                        quan_idx+=1
                        # print("cconv", layer_name, "Kernel:", K, perf[0] / 10 ** 5, perf[1],
                        #       [x / 10 ** 5 for x in perf[2]])

                        sorted_per = torch.tensor(perf[2]).sort()[0]
                        max_lat = sorted_per[-1].item()
                        sec_lat = sorted_per[-2].item()
                        quan_ceil = 17 - int_num
                        quan_floor = min(max(math.floor(16/(float(max_lat)/sec_lat))-int_num,1),quan_ceil-1)


                        quan_count = 6
                        step = max(math.ceil((quan_ceil - quan_floor)/quan_count),1)
                        # print(range(quan_floor,quan_ceil,step))
                        cconv_quan_ss.append(list(range(quan_floor,quan_ceil,step)))
                        cconv_quan_sn.append("Qu")


                    if perf[1] == "computing":
                        # cconv_pattern.append(layer_name)
                        if K not in cconv_pattern.keys():
                            cconv_pattern[K] = [layer_name]
                        else:
                            cconv_pattern[K].append(layer_name)
                        # print(layer_name)
                        # print("cconv",layer_name, "Kernel:", K, perf[0] / 10 ** 5, perf[1], [x / 10 ** 5 for x in perf[2]])

                    if perf[1] == "loading Weight":
                        count[1]+=1
                    elif perf[1] == "loading IFM":
                        count[0]+=1
                    elif perf[1] == "storing OFM":
                        count[2] += 1
                    elif perf[1] == "computing":
                        count[3] += 1
                    else:
                        print(perf[1],"not recognized")
                        sys.exit(0)


            elif T == "dconv":

                # w = model.state_dict()[layer_name + ".weight"]
                # x = max(abs(float(w.min())), abs(float(w.max())))
                # int_num, frac_num = re_quantize(x, 16, True)
                # print('''quan_paras["{}"] = [{}, {}, True]'''.format(layer_name, int_num, frac_num))

                # print("\t",layer_name,M, N, R, C, K, S, T)
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = HW1
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (
                                            HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                                            HW_constraints["r_BRAM"], HW_constraints["r_BRAM_Size"],
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

                    # print(perf[0])

                    dTT+=perf[0]

                    # if perf[1] == "loading Weight":
                    #     w = model.state_dict()[layer_name + ".weight"]
                    #     x = max(abs(float(w.min())), abs(float(w.max())))
                    #     int_num, frac_num = re_quantize(x, 16, True)
                    #     print('''quan_paras["{}"] = [{}, {}, True]'''.format(layer_name, int_num, frac_num))
                    #
                    #
                    if perf[1] == "computing":
                        # print(layer_name)
                        # print("dconv",layer_name, "Kernel:", K, perf[0] / 10 ** 5, perf[1], [x / 10 ** 5 for x in perf[2]])
                        if K not in cconv_pattern.keys():
                            cconv_pattern[K] = [layer_name]
                        else:
                            cconv_pattern[K].append(layer_name)
                    if perf[1] == "loading Weight":
                        count[1]+=1
                    elif perf[1] == "loading IFM":
                        count[0]+=1
                    elif perf[1] == "storing OFM":
                        count[2] += 1
                    elif perf[1] == "computing":
                        count[3] += 1
                    else:
                        print(perf[1],"not recognized")
                        sys.exit(0)
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer,
                                                                                                      nn.AvgPool2d):
            input = layer(input)



    print(len(cconv_quan_ss),cconv_quan_ss)
    print(cconv_quan_sn)

    for k,v in cconv_pattern.items():
        print(k,len(v),v)
    # print(len(cconv_pattern),cconv_pattern)
    # 2 is 200 MHz
    print(cTT,dTT)
    return (cTT+dTT) / 10 ** 5 / 2, count




if __name__== "__main__":

    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='mobilenet_v2'
    )
    parser.add_argument(
        '-c', '--cconv',
        default="160, 12, 32, 32, 3, 10, 10, 10",
        help="hardware desgin of cconv",
    )
    parser.add_argument(
        '-dc', '--dconv',
        default="576, 1, 32, 32, 3, 10, 10, 10",
        help="hardware desgin of cconv",
    )
    parser.add_argument(
        '-d', '--dataset',
        default='imagenet'
    )
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true", )
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset

    if dataset_name == "imagenet":
        if "proxyless" in model_name:
            model = torch.hub.load('mit-han-lab/ProxylessNAS', model_name, pretrained=args.pretrained)
        elif "FBNET" in model_name:
            model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'fbnetc_100')
        else:
            model = globals()[model_name](pretrained=args.pretrained)
    elif dataset_name == "cifar10":
        model = getattr(cifar10_models, model_name)(pretrained=args.pretrained)

    #
    # print(model)
    #
    # for name, para in model.named_parameters():
    #     print(name)

    HW1 = [int(x.strip()) for x in args.dconv.split(",")]
    HW2 = [int(x.strip()) for x in args.cconv.split(",")]


    # print("="*10,model_name,"performance analysis:")
    # print(HW1)
    # print(HW2)
    # print(model)
    total_lat,count = get_performance(model, dataset_name, HW1, HW2)
    # print("=" * 10, model_name, "performance analysis end")
    print(model_name, count, total_lat)

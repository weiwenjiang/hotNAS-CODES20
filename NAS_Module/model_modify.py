from torchvision import models

from torchvision.models import *
from torch import nn
import torch
import sys
import math
sys.path.append("../Interface")
import argparse
from ztNAS_model_change import *
import utils
import bottleneck_conv_only
import bottlenect_conv_dconv
from pattern_generator import *
from search_space import *

def Kernel_Patter(model,layer_names,pattern,args):
    layer_pattern = utils.get_layers_pattern(model, layer_names, pattern, args.device)

    max_p_r = -1

    for k,p in pattern.items():
        p_r = int(p.sum())
        if p_r > max_p_r:
            max_p_r = p_r
    pattern_ones = max_p_r

    for layer_name in layer_names:
        layer = dict(model.named_modules())[layer_name]
        ztNAS_add_kernel_mask(model, layer, layer_name, is_pattern=True,
                          pattern=layer_pattern[layer_name].to(args.device), pattern_ones=pattern_ones)


def Kenel_Quantization(model,layer_names,quan_paras_dict):

    for layer_name in layer_names:
        layer = dict(model.named_modules())[layer_name]
        quan_paras = quan_paras_dict[layer_name]
        ztNAS_add_kernel_quant(model,layer, layer_name, is_quant=True, quan_paras = quan_paras)



def Kenel_Expand(model,layer_kernel_inc,var_k=2):

    for layer_name in layer_kernel_inc:
        layer = dict(model.named_modules())[layer_name]
        ztNAS_modify_kernel_shape(model,layer, layer_name,var_k,increase=True)


#
# layers format:
# [ ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, 480, 512)],
#   ...
# ]


def Channel_Cut(model,layers):

    for layer_pair in layers:
        ofm_cut_layer = layer_pair[0]
        ifm_cut_layer = layer_pair[1]
        bn_cut_layer = layer_pair[2]
        CH0,CH1,CH2 = layer_pair[3]
        force = False
        if len(layer_pair)==5:
            force = layer_pair[4]

        conv_modify = {}
        conv_modify[ofm_cut_layer] = (
            dict(model.named_modules())[ofm_cut_layer], CH0, CH1, [bn_cut_layer, ifm_cut_layer], force)
        conv_modify[ifm_cut_layer] = (dict(model.named_modules())[ifm_cut_layer], CH1, CH2, [])
        bn_modifiy = {}
        bn_modifiy[bn_cut_layer] = (dict(model.named_modules())[bn_cut_layer], CH1)
        ztNAS_cut_channel(model, conv_modify, bn_modifiy)


# [1,22,49,54], 3, [100,210,210,470,470]
def mnasnet0_5_space(model, pattern_3_3_idx, pattern_5_5_idx, q_list, args):

    # pattern_idx = [0, 1, 2, 3]

    pattern_55_space = pattern_sets_generate_3((5, 5))
    pattern_55 = {}
    i = 0
    for idx in pattern_5_5_idx:
        pattern_55[i] = pattern_55_space[idx].reshape((5, 5))
        i+=1
    layer_names_55 = ["layers.9.0.layers.3","layers.9.1.layers.3","layers.9.2.layers.3",
                      "layers.10.1.layers.3","layers.10.2.layers.3"]


    pattern_33_space = pattern_sets_generate_3((3, 3))
    pattern_33 = {}
    i = 0
    for idx in pattern_3_3_idx:
        pattern_33[i] = pattern_33_space[idx].reshape((3, 3))
        i += 1
    # layer_33_names = ["layers.3", "layers.8.1.layers.3", "layers.8.2.layers.3"]
    layer_33_names = ["layers.8.1.layers.3", "layers.8.2.layers.3"]

    Kernel_Patter(model, layer_names_55, pattern_55, args)
    # Kernel_Patter(model, layer_33_names, pattern_33, args)

    quan_paras = {}

    # quan_paras["layers.0"] = [0, q_list[0], True]
    # quan_paras["layers.10.0.layers.3"] = [0, q_list[1], True]
    # quan_paras["layers.12.0.layers.3"] = [0, q_list[2], True]
    # quan_paras["layers.12.0.layers.6"] = [0, q_list[3], True]
    # quan_paras["layers.12.1.layers.0"] = [0, q_list[4], True]
    # quan_paras["layers.12.1.layers.3"] = [0, q_list[5], True]
    # quan_paras["layers.12.1.layers.6"] = [0, q_list[6], True]
    # quan_paras["layers.12.2.layers.0"] = [0, q_list[7], True]
    # quan_paras["layers.12.2.layers.3"] = [0, q_list[8], True]
    # quan_paras["layers.12.2.layers.6"] = [0, q_list[9], True]
    # quan_paras["layers.12.3.layers.0"] = [0, q_list[10], True]
    # quan_paras["layers.12.3.layers.6"] = [0, q_list[11], True]
    # quan_paras["layers.13.0.layers.0"] = [0, q_list[12], True]
    # quan_paras["layers.13.0.layers.6"] = [0, q_list[13], True]
    # quan_paras["layers.14"] = [0, q_list[14], True]
    #
    # channel_cut_layers = [["layers.0", "layers.3", "layers.1", (3, 5, 32)],
    #                       ["layers.3", "layers.6", "layers.4", (5, 5, 16), True],
    #                       # ["layers.0", "layers.3", "layers.1", (3, 16, 32)],
    #                       #
    #                       ]
    #                       # ["layer1.1.conv1", "layer1.1.conv2", "layer1.1.bn1", (64, 64, 64)],
    #                       # ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.bn1", (64, 128, 128)],
    #                       # ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.bn1", (128, ch_list[0], 128)],
    #                       # ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.bn1", (128, ch_list[1], 256)],
    #                       # ["layer3.1.conv1", "layer3.1.conv2", "layer3.1.bn1", (256, ch_list[2], 256)],
    #                       # ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, ch_list[3], 512)],
    #                       # ["layer4.1.conv1", "layer4.1.conv2", "layer4.1.bn1", (512, ch_list[4], 512)]]

    # Channel_Cut(model, channel_cut_layers)

    Kenel_Quantization(model, quan_paras.keys(), quan_paras)

    print(model)
    return model

# [1,22,49,54], 3, [100,210,210,470,470]
def resnet_18_space(model, pattern_idx, k_expand, ch_list, q_list, args):

    parttern_77_space = pattern_sets_generate_3((7, 7))
    parttern_77 = {}
    for i in parttern_77_space.keys():
        parttern_77[i] = parttern_77_space[i].reshape((7, 7))
    layer_names_77 = ["conv1"]

    pattern_space = pattern_sets_generate_3((3, 3))
    pattern = {}
    i = 0
    for idx in pattern_idx:
        pattern[i] = pattern_space[idx].reshape((3, 3))
        i+=1
    layer_names = ["layer1.0.conv1","layer1.0.conv2","layer1.1.conv1",
        "layer1.1.conv2","layer2.0.conv2","layer2.1.conv1","layer2.1.conv2"]


    if k_expand == 0:
        layer_kernel_inc = []
    elif k_expand == 1:
        layer_kernel_inc = ["layer2.0.conv1"]
    elif k_expand == 2:
        layer_kernel_inc = ["layer2.0.downsample.0"]
    else:
        layer_kernel_inc = ["layer2.0.conv1","layer2.0.downsample.0"]

    channel_cut_layers = [["layer1.0.conv1", "layer1.0.conv2", "layer1.0.bn1", (64, 64, 64)],
                          ["layer1.1.conv1", "layer1.1.conv2", "layer1.1.bn1", (64, 64, 64)],
                          ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.bn1", (64, 128, 128)],
                          ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.bn1", (128, ch_list[0], 128)],
                          ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.bn1", (128, ch_list[1], 256)],
                          ["layer3.1.conv1", "layer3.1.conv2", "layer3.1.bn1", (256, ch_list[2], 256)],
                          ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, ch_list[3], 512)],
                          ["layer4.1.conv1", "layer4.1.conv2", "layer4.1.bn1", (512, ch_list[4], 512)]]

    quant_layers = ["layer3.0.conv1", "layer3.0.conv2",
                    "layer3.1.conv1", "layer3.1.conv2",
                    "layer4.0.conv1", "layer4.0.conv2",
                    "layer4.1.conv1", "layer4.1.conv2"]
    quan_paras = {}
    quan_paras["layer3.0.conv1"] = [0, q_list[0], True]
    quan_paras["layer3.0.conv2"] = [0, q_list[1], True]
    quan_paras["layer3.1.conv1"] = [0, q_list[2], True]
    quan_paras["layer3.1.conv2"] = [0, q_list[3], True]
    quan_paras["layer4.0.conv1"] = [0, q_list[4], True]
    quan_paras["layer4.0.conv2"] = [0, q_list[5], True]
    quan_paras["layer4.1.conv1"] = [0, q_list[6], True]
    quan_paras["layer4.1.conv2"] = [0, q_list[7], True]


    Channel_Cut(model, channel_cut_layers)
    Kernel_Patter(model, layer_names, pattern, args)
    Kenel_Expand(model, layer_kernel_inc)
    Kenel_Quantization(model, quant_layers, quan_paras)

    # Kernel_Patter(model, layer_names_77, parttern_77, args)

    return model


def mobilenet_v2_space(model, args):

    quan_paras = {}

    quan_paras["features.0.0"] = [0, 16, True]
    quan_paras["features.1.conv.0.0"] = [0, 16, True]
    quan_paras["features.1.conv.1"] = [1, 15, True]
    quan_paras["features.2.conv.0.0"] = [0, 16, True]
    quan_paras["features.2.conv.1.0"] = [0, 16, True]
    quan_paras["features.2.conv.2"] = [1, 15, True]
    quan_paras["features.3.conv.0.0"] = [0, 16, True]
    quan_paras["features.3.conv.1.0"] = [0, 16, True]
    quan_paras["features.3.conv.2"] = [0, 16, True]
    quan_paras["features.4.conv.0.0"] = [0, 16, True]
    quan_paras["features.4.conv.1.0"] = [0, 16, True]
    quan_paras["features.4.conv.2"] = [0, 16, True]
    quan_paras["features.5.conv.0.0"] = [0, 16, True]
    quan_paras["features.5.conv.1.0"] = [0, 16, True]
    quan_paras["features.5.conv.2"] = [0, 16, True]
    quan_paras["features.6.conv.0.0"] = [0, 16, True]
    quan_paras["features.6.conv.1.0"] = [0, 16, True]
    quan_paras["features.6.conv.2"] = [0, 16, True]
    quan_paras["features.7.conv.0.0"] = [0, 16, True]
    quan_paras["features.7.conv.1.0"] = [0, 16, True]
    quan_paras["features.7.conv.2"] = [0, 16, True]
    quan_paras["features.8.conv.0.0"] = [0, 16, True]
    quan_paras["features.8.conv.1.0"] = [0, 16, True]
    quan_paras["features.8.conv.2"] = [0, 16, True]
    quan_paras["features.9.conv.0.0"] = [0, 16, True]
    quan_paras["features.9.conv.1.0"] = [0, 16, True]
    quan_paras["features.9.conv.2"] = [0, 16, True]
    quan_paras["features.10.conv.0.0"] = [0, 16, True]
    quan_paras["features.10.conv.1.0"] = [0, 16, True]
    quan_paras["features.10.conv.2"] = [0, 16, True]
    quan_paras["features.11.conv.0.0"] = [0, 16, True]
    quan_paras["features.11.conv.1.0"] = [0, 16, True]
    quan_paras["features.11.conv.2"] = [0, 16, True]
    quan_paras["features.12.conv.0.0"] = [0, 16, True]
    quan_paras["features.12.conv.1.0"] = [0, 16, True]
    quan_paras["features.12.conv.2"] = [0, 16, True]
    quan_paras["features.13.conv.0.0"] = [0, 16, True]
    quan_paras["features.13.conv.1.0"] = [0, 16, True]
    quan_paras["features.13.conv.2"] = [0, 16, True]
    quan_paras["features.14.conv.0.0"] = [0, 16, True]
    quan_paras["features.14.conv.1.0"] = [0, 16, True]
    quan_paras["features.14.conv.2"] = [0, 16, True]
    quan_paras["features.15.conv.0.0"] = [0, 16, True]
    quan_paras["features.15.conv.1.0"] = [0, 16, True]
    quan_paras["features.15.conv.2"] = [0, 16, True]
    quan_paras["features.16.conv.0.0"] = [0, 16, True]
    quan_paras["features.16.conv.1.0"] = [0, 16, True]
    quan_paras["features.16.conv.2"] = [0, 16, True]
    quan_paras["features.17.conv.0.0"] = [0, 16, True]
    quan_paras["features.17.conv.1.0"] = [0, 16, True]
    quan_paras["features.17.conv.2"] = [0, 16, True]
    quan_paras["features.18.0"] = [0, 16, True]

    Kenel_Quantization(model, quan_paras.keys(), quan_paras)

    print(model)
    return model


if __name__ == "__main__":
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
        '-d', '--dna',
        # default="8 8 8 8 8 8 8 8 8 8 8 8 8 8 8",
        default="30 39 41 50 130 439 541 250 8 8 8 8 8 8 4 4 4 4 4 4 4 4 4",

        # default="30 39 41 50 0 128 224 224 512 512 4 4 4 4 4 8 16 2 1 -2 2",
        help="exploration results",
    )
    parser.add_argument('--device', default='cpu', help='device')

    args = parser.parse_args()
    model_name = args.model
    model = globals()[model_name]()

    if args.model == "resnet18":
        dna = [int(x) for x in args.dna.split(" ")]
        pat_point, exp_point, ch_point, quant_point, comm_point = dna[0:4], dna[4], dna[5:10], dna[10:18], dna[18:21]
        model = resnet_18_space(model, pat_point, exp_point, ch_point, quant_point, args)

        [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in args.cconv.split(",")]
        print("=" * 10, model_name, "performance analysis:")
        if W_p + comm_point[0] + I_p + comm_point[1] + O_p + comm_point[2] <= int(
                HW_constraints["r_Ports_BW"] / HW_constraints["BITWIDTH"]):
            total_lat = bottleneck_conv_only.get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p + comm_point[0],
                                                             I_p + comm_point[1], O_p + comm_point[2])
            print(total_lat)
        else:
            print("-1")

    elif args.model == "mnasnet0_5":
        dna = [int(x) for x in args.dna.split(" ")]

        pattern_3_3_idx = dna[0:4]
        pattern_5_5_idx = dna[4:8]
        q_list = dna[8:23]
        model = mnasnet0_5_space(model, pattern_3_3_idx, pattern_5_5_idx, q_list, args)
        HW1 = [int(x.strip()) for x in args.dconv.split(",")]
        HW2 = [int(x.strip()) for x in args.cconv.split(",")]


        print("=" * 10, model_name, "performance analysis:")
        total_lat = bottlenect_conv_dconv.get_performance(model, HW1, HW2)

        print(total_lat/2)

    elif args.model == "mobilenet_v2":
        HW1 = [int(x.strip()) for x in args.dconv.split(",")]
        HW2 = [int(x.strip()) for x in args.cconv.split(",")]

        model = mobilenet_v2_space(model, args)
        total_lat = bottlenect_conv_dconv.get_performance(model, HW1, HW2)
        print(total_lat / 2)

    print("Success")


    #
    #
    # sys.exit(0)
    #
    # # print(model)
    #
    # pattern_num = 4
    #
    # device = torch.device(args.device)
    #
    # pattern = {}
    # pattern[0] = torch.tensor([1., 1., 1., 1., 0., 1., 1., 0., 0.])
    # pattern[1] = torch.tensor([0., 0., 1., 1., 1., 1., 1., 0., 1.])
    # pattern[2] = torch.tensor([1., 1., 0., 1., 1., 0., 1., 1., 0.])
    # pattern[3] = torch.tensor([1., 0., 0., 1., 1., 1., 0., 1., 1.])
    #
    # for i in range(pattern_num):
    #     pattern[i] = pattern[i].reshape((3, 3))
    #
    # layer_pattern_train_para = [
    #     "layer1.0.conv1.weight",
    #     "layer1.0.bn1.weight",
    #     "layer1.0.bn1.bias",
    #     "layer1.0.conv2.weight",
    #     "layer1.0.bn2.weight",
    #     "layer1.0.bn2.bias",
    #     "layer1.1.conv1.weight",
    #     "layer1.1.bn1.weight",
    #     "layer1.1.bn1.bias",
    #     "layer1.1.conv2.weight",
    #     "layer1.1.bn2.weight",
    #     "layer1.1.bn2.bias",
    #     "layer2.0.conv2.weight",
    #     "layer2.0.bn2.weight",
    #     "layer2.0.bn2.bias",
    #     "layer2.1.conv1.weight",
    #     "layer2.1.bn1.weight",
    #     "layer2.1.bn1.bias",
    #     "layer2.1.conv2.weight",
    #     "layer2.1.bn2.weight",
    #     "layer2.1.bn2.bias"]
    #
    # layer_names = [
    #     "layer1.0.conv1",
    #     "layer1.0.conv2",
    #     "layer1.1.conv1",
    #     "layer1.1.conv2",
    #     "layer2.0.conv2",
    #     "layer2.1.conv1",
    #     "layer2.1.conv2"
    # ]
    #
    #
    #
    #
    # k_expand = 3
    #
    # if k_expand == 0:
    #     layer_k_expand_train_para = []
    #     layer_kernel_inc = []
    # elif k_expand == 1:
    #     layer_k_expand_train_para = ["layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias"]
    #     layer_kernel_inc = ["layer2.0.conv1"]
    # elif k_expand == 2:
    #     layer_k_expand_train_para = ["layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight",
    #                                  "layer2.0.downsample.1.bias"]
    #     layer_kernel_inc = ["layer2.0.downsample.0"]
    # else:
    #     layer_k_expand_train_para = [
    #         "layer2.0.conv1.weight",
    #         "layer2.0.bn1.weight",
    #         "layer2.0.bn1.bias",
    #         "layer2.0.downsample.0.weight",
    #         "layer2.0.downsample.1.weight",
    #         "layer2.0.downsample.1.bias"]
    #     layer_kernel_inc = [
    #         "layer2.0.conv1",
    #         "layer2.0.downsample.0"
    #     ]
    #
    # layer_train_para = layer_pattern_train_para + layer_k_expand_train_para
    #
    # channel_cut_layers = [["layer1.0.conv1", "layer1.0.conv2", "layer1.0.bn1", (64, 64, 64)],
    #                       ["layer1.1.conv1", "layer1.1.conv2", "layer1.1.bn1", (64, 64, 64)],
    #                       ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.bn1", (64, 128, 128)],
    #                       ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.bn1", (128, 100, 128)],
    #                       ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.bn1", (128, 210, 256)],
    #                       ["layer3.1.conv1", "layer3.1.conv2", "layer3.1.bn1", (256, 210, 256)],
    #                       ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, 470, 512)],
    #                       ["layer4.1.conv1", "layer4.1.conv2", "layer4.1.bn1", (512, 470, 512)]]
    #
    # Channel_Cut(model, channel_cut_layers)
    # Kernel_Patter(model, layer_names, pattern, args)
    # Kenel_Expand(model,layer_kernel_inc)
    #
    # # print("=" * 100)
    # print(model)
    #
    # print("="*40,"Validate function crectness","="*40)
    # input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
    # output = model(input)
    #
    # [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in args.cconv.split(",")]
    # print("=" * 10, model_name, "performance analysis:")
    # total_lat = bottleneck_conv_only.get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p)
    # print(total_lat)
    #
    # print()
    # print("Success")
    #
    #

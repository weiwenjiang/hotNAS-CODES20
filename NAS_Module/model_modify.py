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
from model_search_space.resnet18 import *

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='resnet18'
    )
    parser.add_argument(
        '-c', '--cconv',
        default="160, 12, 32, 32, 7, 10, 10, 10",
        help="hardware desgin of cconv",
    )
    parser.add_argument(
        '-dc', '--dconv',
        default="576, 1, 32, 32, 7, 10, 10, 10",
        help="hardware desgin of cconv",
    )

    parser.add_argument(
        '-d', '--dna',
        # default="8 8 8 8 8 8 8 8 8 8 8 8 8 8 8",
        # default="30 39 41 50 130 439 541 250 8 8 8 8 8 8 4 4 4 4 4 4 4 4 4",

        default="30 39 41 50 0 128 224 224 512 512 4 4 4 4 4 8 16 2 1 -2 2",
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

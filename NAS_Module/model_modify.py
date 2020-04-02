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
from pattern_generator import *

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

        conv_modify = {}
        conv_modify[ofm_cut_layer] = (
            dict(model.named_modules())[ofm_cut_layer], CH0, CH1, [bn_cut_layer, ifm_cut_layer])
        conv_modify[ifm_cut_layer] = (dict(model.named_modules())[ifm_cut_layer], CH1, CH2, [])
        bn_modifiy = {}
        bn_modifiy[bn_cut_layer] = (dict(model.named_modules())[bn_cut_layer], CH1)
        ztNAS_cut_channel(model, conv_modify, bn_modifiy)



# [1,22,49,54], 3, [100,210,210,470,470]
def resnet_18_space(model, pattern_idx, k_expand, ch_list,args):

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

    quant_layers = ["layer4.1.conv1", "layer4.1.conv2"]
    quan_paras = {}
    quan_paras["layer4.1.conv1"] = [0, 2, True]
    quan_paras["layer4.1.conv2"] = [0, 2, True]

    # Channel_Cut(model, channel_cut_layers)
    # Kernel_Patter(model, layer_names, pattern, args)
    # Kenel_Expand(model, layer_kernel_inc)
    Kenel_Quantization(model, quant_layers, quan_paras)

    return model





if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='resnet18'
    )
    parser.add_argument(
        '-c', '--cconv',
        default="70, 36, 64, 64, 7, 20, 6, 6",
        help="hardware desgin of cconv",
    )
    parser.add_argument('--device', default='cpu', help='device')

    args = parser.parse_args()
    model_name = args.model
    model = globals()[model_name]()

    model = resnet_18_space(model, [1, 22, 49, 54], 3, [128, 240, 240, 480, 480], args)

    print(model)

    # [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in args.cconv.split(",")]
    # print("=" * 10, model_name, "performance analysis:")
    # total_lat = bottleneck_conv_only.get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p)
    # print(total_lat)
    #
    # print()
    # print("Success")




    sys.exit(0)

    # print(model)

    pattern_num = 4

    device = torch.device(args.device)

    pattern = {}
    pattern[0] = torch.tensor([1., 1., 1., 1., 0., 1., 1., 0., 0.])
    pattern[1] = torch.tensor([0., 0., 1., 1., 1., 1., 1., 0., 1.])
    pattern[2] = torch.tensor([1., 1., 0., 1., 1., 0., 1., 1., 0.])
    pattern[3] = torch.tensor([1., 0., 0., 1., 1., 1., 0., 1., 1.])

    for i in range(pattern_num):
        pattern[i] = pattern[i].reshape((3, 3))

    layer_pattern_train_para = [
        "layer1.0.conv1.weight",
        "layer1.0.bn1.weight",
        "layer1.0.bn1.bias",
        "layer1.0.conv2.weight",
        "layer1.0.bn2.weight",
        "layer1.0.bn2.bias",
        "layer1.1.conv1.weight",
        "layer1.1.bn1.weight",
        "layer1.1.bn1.bias",
        "layer1.1.conv2.weight",
        "layer1.1.bn2.weight",
        "layer1.1.bn2.bias",
        "layer2.0.conv2.weight",
        "layer2.0.bn2.weight",
        "layer2.0.bn2.bias",
        "layer2.1.conv1.weight",
        "layer2.1.bn1.weight",
        "layer2.1.bn1.bias",
        "layer2.1.conv2.weight",
        "layer2.1.bn2.weight",
        "layer2.1.bn2.bias"]

    layer_names = [
        "layer1.0.conv1",
        "layer1.0.conv2",
        "layer1.1.conv1",
        "layer1.1.conv2",
        "layer2.0.conv2",
        "layer2.1.conv1",
        "layer2.1.conv2"
    ]




    k_expand = 3

    if k_expand == 0:
        layer_k_expand_train_para = []
        layer_kernel_inc = []
    elif k_expand == 1:
        layer_k_expand_train_para = ["layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias"]
        layer_kernel_inc = ["layer2.0.conv1"]
    elif k_expand == 2:
        layer_k_expand_train_para = ["layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight",
                                     "layer2.0.downsample.1.bias"]
        layer_kernel_inc = ["layer2.0.downsample.0"]
    else:
        layer_k_expand_train_para = [
            "layer2.0.conv1.weight",
            "layer2.0.bn1.weight",
            "layer2.0.bn1.bias",
            "layer2.0.downsample.0.weight",
            "layer2.0.downsample.1.weight",
            "layer2.0.downsample.1.bias"]
        layer_kernel_inc = [
            "layer2.0.conv1",
            "layer2.0.downsample.0"
        ]

    layer_train_para = layer_pattern_train_para + layer_k_expand_train_para

    channel_cut_layers = [["layer1.0.conv1", "layer1.0.conv2", "layer1.0.bn1", (64, 64, 64)],
                          ["layer1.1.conv1", "layer1.1.conv2", "layer1.1.bn1", (64, 64, 64)],
                          ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.bn1", (64, 128, 128)],
                          ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.bn1", (128, 100, 128)],
                          ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.bn1", (128, 210, 256)],
                          ["layer3.1.conv1", "layer3.1.conv2", "layer3.1.bn1", (256, 210, 256)],
                          ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, 470, 512)],
                          ["layer4.1.conv1", "layer4.1.conv2", "layer4.1.bn1", (512, 470, 512)]]

    Channel_Cut(model, channel_cut_layers)
    Kernel_Patter(model, layer_names, pattern, args)
    Kenel_Expand(model,layer_kernel_inc)

    # print("=" * 100)
    print(model)

    print("="*40,"Validate function crectness","="*40)
    input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
    output = model(input)

    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in args.cconv.split(",")]
    print("=" * 10, model_name, "performance analysis:")
    total_lat = bottleneck_conv_only.get_performance(model, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p)
    print(total_lat)

    print()
    print("Success")



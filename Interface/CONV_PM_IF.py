from torchvision import models

from torchvision.models import *
from torch import nn
import torch
import sys

sys.path.append("../Performance_Model")
import PM_Config
import PM_Layer
import PM_FPGA_Template

def is_same(shape):
    k_size = shape
    if len(shape) == 2 and shape[0] != shape[1]:
        print("Not support kernel/stride/padding with different size")
        sys.exit(0)
    elif len(shape) == 2:
        k_size = shape[0]
    return k_size

def tell_conv_type(in_channels,groups):
    if in_channels==groups:
        return "dconv"
    elif groups==1:
        return "cconv"
    else:
        print("Not support in_channels!=groups and in_channels!=1")
        sys.exit(0)



Model_Zoo = [ 'alexnet', 'densenet121', 'densenet161',  'densenet169', 'densenet201', 'squeezenet1_0', 'squeezenet1_1',  'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn','wide_resnet101_2', 'wide_resnet50_2',  'vgg11',  'resnet50', 'resnet101', 'resnet152', 'resnet18', 'resnet34' ]

Model_Zoo_w_dconv = [   'mnasnet0_5', 'mobilenet_v2', 'mnasnet1_0', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']

print(len(Model_Zoo))

i = 0
for model_name in Model_Zoo_w_dconv:
    print(model_name)
    model = globals()[model_name]()
    # print(model)

    input = torch.Tensor(torch.Size([1,3,224,224])).to(torch.float32)

    Last_CNN = 0
    First_FC = 0


    cTT = 0
    dTT = 0

    cTT_cmp = 0
    dTT_cmp = 0

    for layer_name,layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            # print("\t=",layer_name)
            input_shape = list(input.shape)
            input_shape[1] = layer.in_channels
            input = torch.Tensor(torch.Size(input_shape)).to(torch.float32)
            # print("\t\tinput:", input.shape)



            # Last_CNN = [layer.out_channels, layer.in_channels, input.shape[2], input.shape[3], is_same(layer.kernel_size),
            # is_same(layer.stride), tell_conv_type(layer.in_channels, layer.groups), is_same(layer.padding)]
            input = layer(input)



            [B, M, N, R, C, K, S, T, P] = (1, layer.out_channels, layer.in_channels, input.shape[2], input.shape[3], is_same(layer.kernel_size),
                  is_same(layer.stride), tell_conv_type(layer.in_channels,layer.groups) , is_same(layer.padding))

            # if layer_name == "features.11":
            #     M = 412
            # if layer_name == "features.13":
            #     M = 412
            #     N = 412
            # if layer_name == "features.16":
            #     N = 412


            #     M = 412
            # if layer_name == "features.18":
            #     N = 412

            # print("\t(M, N, R, C, K, S, T, P):", M, N, R, C, K, S, T, P)

            # if T=="cconv":
            #     [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = (64,32,28,14,3,16,8,16)
            #     [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (1024, 2520, 18000, 1824, 16)
            #     Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv")
            #     acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
            #                            Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)
            #     print("\tcconv:",acc_1.get_layer_latency(Layer))
            #     TT += acc_1.get_layer_latency(Layer)[0]


            #
            # if T=="cconv":
            #     [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = (64,6,14,14,11,28,16,18)
            #     [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (1024, 2520, 18000, 1824, 16)
            #     Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv")
            #     acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
            #                            Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)
            #     # print("\t\tcconv:",acc_1.get_layer_latency(Layer))
            #     cTT += acc_1.get_layer_latency(Layer)[0]
            # elif T=="dconv":
            #     [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = (64,32,14,14,5,6,12,6)
            #     [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (1024, 2520, 18000, 1824, 16)
            #     Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "dconv")
            #     acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
            #                            Tk, W_p, I_p, O_p, "dconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)
            #     # print("\t\tdconv:",acc_1.get_layer_latency(Layer))
            #     dTT += acc_1.get_layer_latency(Layer)[0]
            #

            if T=="cconv":
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = (64,34,14,14,11,28,16,18)
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (1024, 2520, 18000, 1824, 16)
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv")
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                       Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)
                # print("\t\tcconv:",acc_1.get_layer_latency(Layer))
                cTT_cmp += acc_1.get_layer_latency(Layer)[0]
            elif T=="dconv":
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = (128,1,14,14,5,6,12,6)
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (1024, 2520, 18000, 1824, 16)
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "dconv")
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                       Tk, W_p, I_p, O_p, "dconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)
                # print("\t\tdconv:",acc_1.get_layer_latency(Layer))
                dTT_cmp += acc_1.get_layer_latency(Layer)[0]


        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d)  or isinstance(layer, nn.AvgPool2d):
            input = layer(input)

    # print("\tTotal Time:", (cTT + dTT) / 10 ** 5)
    # print("\tTime cconv:", (cTT) / 10 ** 5)
    # print("\tTime dconv:", (dTT) / 10 ** 5)

    print("\tTotal Time:", (cTT_cmp + dTT_cmp) / 10 ** 5)

    # i+=1
    # if i==1:
    #     print("Total Time:",(cTT+dTT)/10**5)
    #     print("\tTime cconv:", (cTT) / 10 ** 5)
    #     print("\tTime dconv:", (dTT) / 10 ** 5)
    #
    #     print("="*50)
    #
    #     print("Total Time:", (cTT_cmp + dTT_cmp) / 10 ** 5)
    #     print("\tTime cconv:", (cTT_cmp) / 10 ** 5)
    #     print("\tTime dconv:", (dTT_cmp) / 10 ** 5)
    #     sys.exit(0)
    #



sys.exit(0)















for model_name in Model_Zoo:
    print(model_name)
    model = globals()[model_name]()
    # print(model)

    input = torch.Tensor(torch.Size([1,3,224,224])).to(torch.float32)

    Last_CNN = 0
    First_FC = 0


    cTT = 0
    dTT = 0

    for layer_name,layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            # print("\t=",layer_name)
            input_shape = list(input.shape)
            input_shape[1] = layer.in_channels
            input = torch.Tensor(torch.Size(input_shape)).to(torch.float32)
            # print("\t\tinput:", input.shape)


            input = layer(input)



            [B, M, N, R, C, K, S, T, P] = (1, layer.out_channels, layer.in_channels, input.shape[2], input.shape[3], is_same(layer.kernel_size),
                  is_same(layer.stride), tell_conv_type(layer.in_channels,layer.groups) , is_same(layer.padding))

            if T=="cconv":
                [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = (64,32,28,14,11,16,8,16)
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (1024, 2520, 18000, 1824, 16)
                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv")
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                       Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)
                # print("\tcconv:",acc_1.get_layer_latency(Layer))
                cTT += acc_1.get_layer_latency(Layer)[0]



        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d)  or isinstance(layer, nn.AvgPool2d):
            input = layer(input)

    print("\tTotal Time:", (cTT + dTT) / 10 ** 5)
from torchvision import models

from torchvision.models import *
from torch import nn
import torch
import sys

sys.path.append("../Performance_Model")
import PM_Config
import PM_Layer
import PM_FPGA_Template

from ztNAS_model_change import *
from utility import *


def tell_conv_type(in_channels,groups):
    if in_channels==groups:
        return "dconv"
    elif groups==1:
        return "cconv"
    else:
        print("Not support in_channels!=groups and in_channels!=1")
        sys.exit(0)

# def access_by_name(model,name):
#     seq = name.split(".")
#     a = model
#     for var in seq:
#         if var.isdigit():
#             a = a[int(var)]
#         else:
#             a = getattr(a, var)
#     return a
#



if __name__== "__main__":

    Model_Zoo = [ 'resnet18', 'densenet121', 'alexnet',  'densenet161',  'densenet169', 'densenet201', 'squeezenet1_0', 'squeezenet1_1',  'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn','wide_resnet101_2', 'wide_resnet50_2',  'vgg11',  'resnet50', 'resnet101', 'resnet152',  'resnet34' ]

    Model_Zoo_w_dconv = [   'mobilenet_v2', 'mnasnet0_5',  'mnasnet1_0', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']

    print(len(Model_Zoo))

    i = 0
    for model_name in Model_Zoo:
        print(model_name)
        model = globals()[model_name]()
        print(model)

        input = torch.Tensor(torch.Size([1,3,224,224])).to(torch.float32)

        Last_CNN = 0
        First_FC = 0


        cTT = 0
        dTT = 0

        cTT_cmp = 0
        dTT_cmp = 0




        for layer_name,layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                print(layer_name)





                ztNAS_modify_kernel(model, layer, layer_name, 2)





                sys.exit(0)

                input_shape = list(input.shape)
                input_shape[1] = layer.in_channels
                input = torch.Tensor(torch.Size(input_shape)).to(torch.float32)
                input = layer(input)







            elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d)  or isinstance(layer, nn.AvgPool2d):
                input = layer(input)



        print(model)
        sys.exit(0)






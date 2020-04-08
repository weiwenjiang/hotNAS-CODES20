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
from copy_conv2d import *

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def quantize(x, num_int_bits, num_frac_bits, signed=True):
    precision = 1 / 2 ** num_frac_bits
    x = torch.round(x / precision) * precision

    if signed is True:
        bound = 2 ** (num_int_bits - 1)
        return torch.clamp(x, -bound, bound - precision)
    else:
        bound = 2 ** num_int_bits
        return torch.clamp(x, 0, bound - precision)


if __name__== "__main__":

    # x = torch.tensor(2.634,dtype=torch.float32)
    # print(quantize(x,3,32,True))
    #
    # sys.exit(0)

    # B, C, H, W = 10, 3, 4, 4
    # x = torch.randn(B, C, H, W)
    # y = torch.where(x > x.view(B, C, -1).mean(2)[:, :, None, None], torch.tensor([1.]), torch.tensor([0.]))
    #
    # print(x.shape)
    # print(y.shape)
    # sys.exit(0)
    #



    Model_Zoo = [ 'resnet18', 'densenet121', 'alexnet',  'densenet161',  'densenet169', 'densenet201', 'squeezenet1_0', 'squeezenet1_1',  'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn','wide_resnet101_2', 'wide_resnet50_2',  'vgg11',  'resnet50', 'resnet101', 'resnet152',  'resnet34' ]

    Model_Zoo_w_dconv = [   'mobilenet_v2', 'mnasnet0_5',    'mnasnet1_0', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']

    print(len(Model_Zoo))

    i = 0
    for model_name in Model_Zoo_w_dconv:
        print(model_name)
        model = globals()[model_name](pretrained=True)
        print(model)

        for name, param in model.named_parameters():
            if max(abs(float(param.min())),abs(float(param.max()))) > 1:
                print (name,param.requires_grad,param.data.shape,float(param.min()),float(param.max()))

        sys.exit(0)
        continue

        np = model.named_parameters()

        print(np)
        sys.exit(0)



        input = torch.Tensor(torch.Size([1,3,224,224])).to(torch.float32)

        Last_CNN = 0
        First_FC = 0


        cTT = 0
        dTT = 0

        cTT_cmp = 0
        dTT_cmp = 0


        layer_names = []
        for layer_name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                if is_same(layer.kernel_size) == 3 and layer.in_channels == 512 and layer.out_channels == 512:
                    layer_names.append(layer_name)

        for name, param in model.named_parameters():
            names = [n+"." for n in name.split(".")[:-1]]
            if "".join(names)[:-1] not in layer_names:
                param.requires_grad = False

        # for name, param in model.named_parameters():
        #     print (name,param.requires_grad,param.data.shape)

        global_input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
        model.eval()
        A = model(global_input)


        for layer_name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                # print(layer_name)
                # if is_same(layer.kernel_size) == 3 and layer.in_channels == 512 and layer.out_channels == 512:
                if is_same(layer.kernel_size) == 3:
                    # print(layer_name, layer)

                    pattern3_3 = {}
                    pattern3_3[0] = torch.tensor([[0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float32, device=device)
                    pattern3_3[1] = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.float32, device=device)
                    pattern3_3[2] = torch.tensor([[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=torch.float32, device=device)
                    pattern3_3[3] = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=device)

                    weight_shape = model.state_dict()[layer_name + ".weight"][:].shape
                    shape = list(model.state_dict()[layer_name + ".weight"][:].shape[:-2])
                    shape.append(1)
                    shape.append(1)
                    after_pattern_0 = model.state_dict()[layer_name + ".weight"][:]*pattern3_3[0]
                    after_norm_0 = after_pattern_0.norm(dim=(2,3)).reshape(shape)
                    after_pattern_1 = model.state_dict()[layer_name + ".weight"][:] * pattern3_3[1]
                    after_norm_1 = after_pattern_1.norm(dim=(2, 3)).reshape(shape)
                    after_pattern_2 = model.state_dict()[layer_name + ".weight"][:] * pattern3_3[2]
                    after_norm_2 = after_pattern_2.norm(dim=(2, 3)).reshape(shape)
                    after_pattern_3 = model.state_dict()[layer_name + ".weight"][:] * pattern3_3[3]
                    after_norm_3 = after_pattern_3.norm(dim=(2, 3)).reshape(shape)


                    max_norm = (torch.max(torch.max(torch.max(after_norm_0,after_norm_1),after_norm_2),after_norm_3))
                    pattern = torch.zeros_like(model.state_dict()[layer_name + ".weight"][:])

                    pattern = pattern + (after_norm_0==max_norm).float()*pattern3_3[0] + \
                              (after_norm_1==max_norm).float()* pattern3_3[1] + \
                              (after_norm_2==max_norm).float()* pattern3_3[2] +\
                              (after_norm_3==max_norm).float()* pattern3_3[3]

                    # weight = model.state_dict()[layer_name + ".weight"][:] * pattern
                    #
                    # print(weight)

                    ztNAS_add_kernel_mask(model, layer, layer_name, is_pattern=True, pattern=pattern)




                    # new_after_norm_0 = after_norm_0.clone()
                    # new_after_norm_0.reshape(weight_shape)
                    # print(new_after_norm_0.shape)
                    # # new_after_norm_0[after_norm_0 == max_norm,:,:] = pattern3_3[0]
                    # # print(new_after_norm_0)

                    # print(model.state_dict()[layer_name + ".weight"][:]*pattern3_3[0])
                    # print(after_pattern_0.shape)
                    # print((after_norm_0==max_norm).int().shape)
                    # print(after_pattern_0[after_norm_0==max_norm].shape)




                    # sys.exit(0)

                    #
                    #
                    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.float32, device=device)
                    # ztNAS_add_kernel_mask(model, layer, layer_name, mask=mask)

                # else:
                #     ztNAS_modify_kernel_shape(model, layer, layer_name, var_k=2)
                #

            #     print(model)
            #
            #
            #
            #     sys.exit(0)
            #
            #     input_shape = list(input.shape)
            #     input_shape[1] = layer.in_channels
            #     input = torch.Tensor(torch.Size(input_shape)).to(torch.float32)
            #     input = layer(input)
            #
            #
            #
            #
            #
            #
            #
            # elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d)  or isinstance(layer, nn.AvgPool2d):
            #     input = layer(input)
            #

        model.to(device)
        model.eval()
        # print(model)
        B = model(global_input)



        sys.exit(0)




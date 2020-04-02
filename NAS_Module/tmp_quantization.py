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

    for name,layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            print(name)
            print("\t",float(model.state_dict()[name+".weight"].max()),float(model.state_dict()[name+".weight"].min()))



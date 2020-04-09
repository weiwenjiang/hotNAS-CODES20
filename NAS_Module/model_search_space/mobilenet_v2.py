import sys
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
from model_modify import *


def mobilenet_v2_space(model, args):

    quan_paras = {}

    quan_paras["features.0.0"] = [2, 14, True]
    quan_paras["features.14.conv.2"] = [1, 15, True]
    quan_paras["features.15.conv.0.0"] = [1, 15, True]
    quan_paras["features.15.conv.2"] = [1, 15, True]
    quan_paras["features.16.conv.0.0"] = [1, 15, True]
    quan_paras["features.16.conv.2"] = [1, 15, True]
    quan_paras["features.17.conv.0.0"] = [1, 15, True]
    quan_paras["features.17.conv.2"] = [1, 15, True]
    quan_paras["features.18.0"] = [2, 14, True]

    Kenel_Quantization(model, quan_paras.keys(), quan_paras)

    print(model)
    return model

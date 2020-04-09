import sys
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
from model_modify import *



#
# layers format:
# [ ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, 480, 512)],
#   ...
# ]



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




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='resnet18'
    )
    parser.add_argument(
        '-c', '--cconv',
        default="70, 36, 64, 64, 7, 18, 6, 6",
        help="hardware desgin of cconv",
    )
    parser.add_argument(
        '-dc', '--dconv',
        default="576, 1, 32, 32, 3, 10, 10, 10",
        help="hardware desgin of cconv",
    )

    parser.add_argument(
        '-d', '--dna',
        default="30 39 41 50 0 128 224 224 512 512 4 4 4 4 4 8 16 2 1 -2 2",
        help="exploration results",
    )
    parser.add_argument('--device', default='cpu', help='device')

    args = parser.parse_args()
    model_name = args.model
    model = globals()[model_name]()

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

    print("Success")

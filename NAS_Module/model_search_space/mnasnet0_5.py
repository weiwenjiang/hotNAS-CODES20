import sys
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
from model_modify import *


# [1,22,49,54], 3, [100,210,210,470,470]
def mnasnet0_5_space(model, pattern_3_3_idx, pattern_5_5_idx, q_list, args):

    # pattern_idx = [0, 1, 2, 3]

    pattern_55_space = pattern_sets_generate_3((5, 5),6)
    pattern_55 = {}
    i = 0
    for idx in pattern_5_5_idx:
        pattern_55[i] = pattern_55_space[idx].reshape((5, 5))
        i+=1
    layer_names_55 = ["layers.9.0.layers.3","layers.9.1.layers.3","layers.9.2.layers.3",
                      "layers.10.1.layers.3","layers.10.2.layers.3"]


    pattern_33_space = pattern_sets_generate_3((3, 3), 1)
    pattern_33 = {}
    i = 0
    for idx in pattern_3_3_idx:
        pattern_33[i] = pattern_33_space[idx].reshape((3, 3))
        i += 1

    layer_33_names = ["layers.3", "layers.8.1.layers.3", "layers.8.2.layers.3"]

    Kernel_Patter(model, layer_names_55, pattern_55, args)
    Kernel_Patter(model, layer_33_names, pattern_33, args)

    # Change all layer to 16 bit
    quan_paras = {}

    quan_paras["layers.0"] = [6, 10, True]
    quan_paras["layers.3"] = [6, 10, True]
    quan_paras["layers.6"] = [5, 11, True]
    quan_paras["layers.8.0.layers.0"] = [5, 11, True]
    quan_paras["layers.8.0.layers.3"] = [5, 11, True]
    quan_paras["layers.8.0.layers.6"] = [5, 11, True]
    quan_paras["layers.8.1.layers.0"] = [5, 11, True]
    quan_paras["layers.8.1.layers.3"] = [5, 11, True]
    quan_paras["layers.8.1.layers.6"] = [4, 12, True]
    quan_paras["layers.8.2.layers.0"] = [5, 11, True]
    quan_paras["layers.8.2.layers.3"] = [5, 11, True]
    quan_paras["layers.8.2.layers.6"] = [4, 12, True]
    quan_paras["layers.9.0.layers.0"] = [5, 11, True]
    quan_paras["layers.9.0.layers.3"] = [4, 12, True]
    quan_paras["layers.9.0.layers.6"] = [5, 11, True]
    quan_paras["layers.9.1.layers.0"] = [4, 12, True]
    quan_paras["layers.9.1.layers.3"] = [5, 11, True]
    quan_paras["layers.9.1.layers.6"] = [4, 12, True]
    quan_paras["layers.9.2.layers.0"] = [4, 12, True]
    quan_paras["layers.9.2.layers.3"] = [5, 11, True]
    quan_paras["layers.9.2.layers.6"] = [4, 12, True]
    quan_paras["layers.10.0.layers.0"] = [5, 11, True]
    quan_paras["layers.10.0.layers.3"] = [4, 12, True]
    quan_paras["layers.10.0.layers.6"] = [4, 12, True]
    quan_paras["layers.10.1.layers.0"] = [4, 12, True]
    quan_paras["layers.10.1.layers.3"] = [4, 12, True]
    quan_paras["layers.10.1.layers.6"] = [4, 12, True]
    quan_paras["layers.10.2.layers.0"] = [4, 12, True]
    quan_paras["layers.10.2.layers.3"] = [4, 12, True]
    quan_paras["layers.10.2.layers.6"] = [4, 12, True]
    quan_paras["layers.11.0.layers.0"] = [4, 12, True]
    quan_paras["layers.11.0.layers.3"] = [5, 11, True]
    quan_paras["layers.11.0.layers.6"] = [4, 12, True]
    quan_paras["layers.11.1.layers.0"] = [4, 12, True]
    quan_paras["layers.11.1.layers.3"] = [4, 12, True]
    quan_paras["layers.11.1.layers.6"] = [4, 12, True]
    quan_paras["layers.12.0.layers.0"] = [4, 12, True]
    quan_paras["layers.12.0.layers.3"] = [3, 13, True]
    quan_paras["layers.12.0.layers.6"] = [4, 12, True]
    quan_paras["layers.12.1.layers.0"] = [3, 13, True]
    quan_paras["layers.12.1.layers.3"] = [4, 12, True]
    quan_paras["layers.12.1.layers.6"] = [3, 13, True]
    quan_paras["layers.12.2.layers.0"] = [4, 12, True]
    quan_paras["layers.12.2.layers.3"] = [5, 11, True]
    quan_paras["layers.12.2.layers.6"] = [3, 13, True]
    quan_paras["layers.12.3.layers.0"] = [4, 12, True]
    quan_paras["layers.12.3.layers.3"] = [4, 12, True]
    quan_paras["layers.12.3.layers.6"] = [3, 13, True]
    quan_paras["layers.13.0.layers.0"] = [4, 12, True]
    quan_paras["layers.13.0.layers.3"] = [4, 12, True]
    quan_paras["layers.13.0.layers.6"] = [3, 13, True]
    quan_paras["layers.14"] = [3, 13, True]

    Kenel_Quantization(model, quan_paras.keys(), quan_paras)

    # Modify layers that is dominated by loading weight
    quan_paras = {}
    quan_paras["layers.0"] = [6, 4, True]
    quan_paras["layers.10.0.layers.3"] = [4, 4, True]
    quan_paras["layers.12.0.layers.3"] = [3, 2, True]
    quan_paras["layers.12.0.layers.6"] = [4, 1, True]
    quan_paras["layers.12.1.layers.0"] = [3, 1, True]
    quan_paras["layers.12.1.layers.3"] = [4, 4, True]
    quan_paras["layers.12.1.layers.6"] = [3, 2, True]
    quan_paras["layers.12.2.layers.0"] = [4, 1, True]
    quan_paras["layers.12.2.layers.3"] = [5, 4, True]
    quan_paras["layers.12.2.layers.6"] = [3, 1, True]
    quan_paras["layers.12.3.layers.0"] = [4, 2, True]
    quan_paras["layers.12.3.layers.3"] = [4, 4, True]
    quan_paras["layers.12.3.layers.6"] = [3, 1, True]
    quan_paras["layers.13.0.layers.0"] = [4, 1, True]
    quan_paras["layers.13.0.layers.6"] = [3, 1, True]
    quan_paras["layers.14"] = [3, 1, True]

    Kenel_Quantization(model, quan_paras.keys(), quan_paras)



    # layer_name = "layers.12.3.layers.3"
    # seq = layer_name.split(".")
    # (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)
    # print(last_attr[3].check_layer())

    # print(model)
    return model




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-m', '--model',
        default='mnasnet0_5'
    )
    parser.add_argument(
        '-c', '--cconv',
        default="100, 16, 32, 32, 3, 6, 10, 14",
        help="hardware desgin of cconv",
    )
    parser.add_argument(
        '-dc', '--dconv',
        default="832, 1, 32, 32, 5, 6, 10, 14",
        help="hardware desgin of cconv",
    )

    parser.add_argument(
        '-d', '--dna',
        # default="8 8 8 8 8 8 8 8 8 8 8 8 8 8 8",
        default="0 1 2 3 130 439 342 250 8 8 8 8 8 8 4 4 4 4 4 4 4 4 4",

        # default="30 39 41 50 0 128 224 224 512 512 4 4 4 4 4 8 16 2 1 -2 2",
        help="exploration results",
    )
    parser.add_argument('--device', default='cpu', help='device')

    args = parser.parse_args()
    model_name = args.model
    model = globals()[model_name]()

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


    print("Success")

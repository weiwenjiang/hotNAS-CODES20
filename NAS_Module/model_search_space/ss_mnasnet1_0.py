import sys
from torchvision import models

from torchvision.models import *
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
from model_modify import *
import model_modify
import random


import train
import random
import time
import datetime
# [1,22,49,54], 3, [100,210,210,470,470]
def mnasnet1_0_space(model, dna, hw_cconv, hw_dconv, args):
    global p3size
    global p5size

    pattern_3_3_idx = dna[0:4]
    pattern_5_5_idx = dna[4:8]
    pattern_do_or_not = dna[8:15]
    q_list = dna[15:25]
    hw_port = dna[25:]

    hw_cconv[5] += hw_port[0]
    hw_cconv[6] += hw_port[1]
    hw_cconv[7] = 32 - hw_cconv[5] - hw_cconv[6]

    hw_dconv[5],hw_dconv[6],hw_dconv[7] = hw_cconv[5],hw_cconv[6],hw_cconv[7]
    # pattern_idx = [0, 1, 2, 3]

    pattern_55_space = pattern_sets_generate_3((5, 5), p5size)

    pattern_55 = {}
    i = 0
    for idx in pattern_5_5_idx:
        pattern_55[i] = pattern_55_space[idx].reshape((5, 5))
        i+=1
    layer_names_55 = ["layers.9.0.layers.3","layers.9.1.layers.3","layers.9.2.layers.3"]

    layer_names_55_select = []
    for i in range(len(layer_names_55)):
        if pattern_do_or_not[i] == 1:
            layer_names_55_select.append(layer_names_55[i])

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)

    pattern_33 = {}
    i = 0
    for idx in pattern_3_3_idx:
        pattern_33[i] = pattern_33_space[idx].reshape((3, 3))
        i += 1

    layer_33_names = ["layers.0","layers.3","layers.8.1.layers.3","layers.8.2.layers.3"]

    layer_names_33_select = []
    for i in range(len(layer_33_names)):
        if pattern_do_or_not[i+len(layer_names_55)] == 1:
            layer_names_33_select.append(layer_33_names[i])

    model_modify.Kernel_Patter(model, layer_names_55_select, pattern_55, args)
    model_modify.Kernel_Patter(model, layer_names_33_select, pattern_33, args)

    # Change all layer to 16 bit
    quan_paras = {}

    quan_paras["layers.0"] = [3, 13, True]
    quan_paras["layers.3"] = [4, 12, True]
    quan_paras["layers.6"] = [3, 13, True]
    quan_paras["layers.8.0.layers.0"] = [3, 13, True]
    quan_paras["layers.8.0.layers.3"] = [2, 14, True]
    quan_paras["layers.8.0.layers.6"] = [3, 13, True]
    quan_paras["layers.8.1.layers.0"] = [3, 13, True]
    quan_paras["layers.8.1.layers.3"] = [2, 14, True]
    quan_paras["layers.8.1.layers.6"] = [2, 14, True]
    quan_paras["layers.8.2.layers.0"] = [2, 14, True]
    quan_paras["layers.8.2.layers.3"] = [3, 13, True]
    quan_paras["layers.8.2.layers.6"] = [2, 14, True]
    quan_paras["layers.9.0.layers.0"] = [3, 13, True]
    quan_paras["layers.9.0.layers.3"] = [2, 14, True]
    quan_paras["layers.9.0.layers.6"] = [2, 14, True]
    quan_paras["layers.9.1.layers.0"] = [1, 15, True]
    quan_paras["layers.9.1.layers.3"] = [2, 14, True]
    quan_paras["layers.9.1.layers.6"] = [2, 14, True]
    quan_paras["layers.9.2.layers.0"] = [2, 14, True]
    quan_paras["layers.9.2.layers.3"] = [2, 14, True]
    quan_paras["layers.9.2.layers.6"] = [2, 14, True]
    quan_paras["layers.10.0.layers.0"] = [2, 14, True]
    quan_paras["layers.10.0.layers.3"] = [1, 15, True]
    quan_paras["layers.10.0.layers.6"] = [2, 14, True]
    quan_paras["layers.10.1.layers.0"] = [1, 15, True]
    quan_paras["layers.10.1.layers.3"] = [2, 14, True]
    quan_paras["layers.10.1.layers.6"] = [1, 15, True]
    quan_paras["layers.10.2.layers.0"] = [2, 14, True]
    quan_paras["layers.10.2.layers.3"] = [2, 14, True]
    quan_paras["layers.10.2.layers.6"] = [2, 14, True]
    quan_paras["layers.11.0.layers.0"] = [2, 14, True]
    quan_paras["layers.11.0.layers.3"] = [2, 14, True]
    quan_paras["layers.11.0.layers.6"] = [2, 14, True]
    quan_paras["layers.11.1.layers.0"] = [1, 15, True]
    quan_paras["layers.11.1.layers.3"] = [2, 14, True]
    quan_paras["layers.11.1.layers.6"] = [1, 15, True]
    quan_paras["layers.12.0.layers.0"] = [1, 15, True]
    quan_paras["layers.12.0.layers.3"] = [1, 15, True]
    quan_paras["layers.12.0.layers.6"] = [1, 15, True]
    quan_paras["layers.12.1.layers.0"] = [1, 15, True]
    quan_paras["layers.12.1.layers.3"] = [2, 14, True]
    quan_paras["layers.12.1.layers.6"] = [1, 15, True]
    quan_paras["layers.12.2.layers.0"] = [1, 15, True]
    quan_paras["layers.12.2.layers.3"] = [2, 14, True]
    quan_paras["layers.12.2.layers.6"] = [1, 15, True]
    quan_paras["layers.12.3.layers.0"] = [1, 15, True]
    quan_paras["layers.12.3.layers.3"] = [2, 14, True]
    quan_paras["layers.12.3.layers.6"] = [1, 15, True]
    quan_paras["layers.13.0.layers.0"] = [1, 15, True]
    quan_paras["layers.13.0.layers.3"] = [1, 15, True]
    quan_paras["layers.13.0.layers.6"] = [1, 15, True]
    quan_paras["layers.14"] = [1, 15, True]

    model_modify.Kenel_Quantization(model, quan_paras.keys(), quan_paras)



    # Modify layers that is dominated by loading weight
    quan_paras = {}
    quan_paras["layers.12.0.layers.6"] = [1, q_list[0], True]
    quan_paras["layers.12.1.layers.0"] = [1, q_list[1], True]
    quan_paras["layers.12.1.layers.6"] = [1, q_list[2], True]
    quan_paras["layers.12.2.layers.0"] = [1, q_list[3], True]
    quan_paras["layers.12.2.layers.6"] = [1, q_list[4], True]
    quan_paras["layers.12.3.layers.0"] = [1, q_list[5], True]
    quan_paras["layers.12.3.layers.6"] = [1, q_list[6], True]
    quan_paras["layers.13.0.layers.0"] = [1, q_list[7], True]
    quan_paras["layers.13.0.layers.6"] = [1, q_list[8], True]
    quan_paras["layers.14"] = [1, q_list[9], True]

    model_modify.Kenel_Quantization(model, quan_paras.keys(), quan_paras)



    # layer_name = "layers.12.3.layers.3"
    # seq = layer_name.split(".")
    # (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)
    # print(last_attr[3].check_layer())

    # print(model)
    return model, hw_cconv, hw_dconv

def get_space():
    global p3size
    global p5size

    p3size = 2
    p5size = 8

    space_name = ("KP-3","KP-3","KP-3","KP-3",
                  "KP-5","KP-5","KP-5","KP-5",
                  "KP5 S or N", "KP5 S or N", "KP5 S or N",
                  "KP3 S or N", "KP3 S or N", "KP3 S or N", "KP3 S or N",
                  "Quan","Quan","Quan","Quan",
                  "Quan","Quan","Quan","Quan",
                  "Quan","Quan",
                  "I_p", "W_p")

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    pattern_55_space = pattern_sets_generate_3((5, 5), p5size)
    k3_s = len(pattern_33_space.keys())
    k5_s = len(pattern_55_space.keys())
    space = (list(range(k3_s)),list(range(k3_s)),list(range(k3_s)),list(range(k3_s)),
             list(range(k5_s)), list(range(k5_s)), list(range(k5_s)), list(range(k5_s)),
             [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
             list(range(4, 15, 2)), list(range(4, 15, 2)), list(range(4, 15, 2)), list(range(4, 15, 2)),
             list(range(4, 15, 2)), list(range(4, 15, 2)), list(range(4, 15, 2)), list(range(4, 15, 2)),
             list(range(4, 15, 2)), list(range(4, 15, 2)),
             [-3,-2,-1,0,1,2,3], [-3,-2,-1,0,1,2,3])
    return space_name,space

def dna_analysis(dna,logger):
    global p3size
    global p5size

    pattern_3_3_idx = dna[0:4]
    pattern_5_5_idx = dna[4:8]
    pattern_do_or_not = dna[8:15]
    q_list = dna[15:25]
    hw_port = dna[25:]

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    pattern_55_space = pattern_sets_generate_3((5, 5), p5size)

    for p in pattern_3_3_idx:
        logger.info("--------->Pattern 3-3 {}: {}".format(p, pattern_33_space[p].flatten()))
    for p in pattern_5_5_idx:
        logger.info("--------->Pattern 5-5 {}: {}".format(p, pattern_55_space[p].flatten()))
    logger.info("--------->Weight Pruning or Not: {}".format(pattern_do_or_not))
    logger.info("--------->Quantization Selection: {}".format(q_list))
    logger.info("--------->Hardware Modification: {}".format(hw_port))




if __name__ == "__main__":

    local = False

    if local:

        parser = argparse.ArgumentParser('Parser User Input Arguments')
        parser.add_argument(
            '-m', '--model',
            default='mnasnet1_0'
        )
        parser.add_argument(
            '-c', '--cconv',
            default="100, 18, 32, 32, 3, 10, 12, 10",
            help="hardware desgin of cconv",
        )
        parser.add_argument(
            '-dc', '--dconv',
            default="704, 1, 32, 32, 5, 10, 12, 10",
            help="hardware desgin of cconv",
        )

        parser.add_argument('--device', default='cpu', help='device')
        args = parser.parse_args()

        model_name = "mnasnet1_0"
        model = globals()["mnasnet1_0"]()
        HW1 = [int(x.strip()) for x in args.dconv.split(",")]
        HW2 = [int(x.strip()) for x in args.cconv.split(",")]



        count = 100
        latency = []

        for i in range(count):

            _, space = get_space()
            dna = []
            for selection in space:
                dna.append(random.choice(selection))
            print(dna)

            import copy
            HW1_c = copy.deepcopy(HW1)
            HW2_c = copy.deepcopy(HW2)

            model,HW_cconv,HW_dconv = mnasnet1_0_space(model, dna, HW2_c, HW1_c, args)
            # model = model.to(args.device)
            print("=" * 10, model_name, "performance analysis:")
            total_lat = bottlenect_conv_dconv.get_performance(model, HW_dconv, HW_cconv, args.device)
            print(total_lat)
            latency.append(total_lat)
            print("=" * 100)

        print("=" * 100)

        print(min(latency),max(latency),sum(latency)/len(latency))


    else:


        args = train.parse_args()
        data_loader, data_loader_test = train.get_data_loader(args)

        model_name = "mnasnet1_0"
        model = globals()["mnasnet1_0"]()
        model = model.to(args.device)
        HW1 = [int(x.strip()) for x in args.dconv.split(",")]
        HW2 = [int(x.strip()) for x in args.cconv.split(",")]

        start_time = time.time()
        count = 150

        latency = []
        record = {}
        for i in range(count):

            _, space = get_space()
            dna = []
            for selection in space:
                dna.append(random.choice(selection))
            print(dna)

            # pattern_3_3_idx = dna[0:4]
            # pattern_5_5_idx = dna[4:8]
            # pattern_do_or_not = dna[8:16]
            # q_list = dna[16:]

            import copy

            HW1_c = copy.deepcopy(HW1)
            HW2_c = copy.deepcopy(HW2)

            model, HW_cconv, HW_dconv = mnasnet1_0_space(model, dna, HW2_c, HW1_c, args)

            # model = mnasnet1_0_space(model, dna, args)
            model = model.to(args.device)
            print("=" * 10, model_name, "performance analysis:")
            total_lat = bottlenect_conv_dconv.get_performance(model, HW_dconv, HW_cconv, args.device)
            print(total_lat)
            latency.append(total_lat)

            acc1, acc5, _ = train.main(args, dna, HW_cconv, data_loader, data_loader_test, HW_dconv)
            print(acc1, acc5, total_lat)
            record[i] = (acc5, total_lat)
            print("Random {}: acc-{}, lat-{}".format(i, acc5, total_lat))
            print(dna)
            print("=" * 100)

        print("=" * 100)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print("Exploration End, using time {}".format(total_time_str))
        for k, v in record.items():
            print(k, v)

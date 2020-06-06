import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
import cifar10_models
from model_modify import *
import model_modify
import random
import train
import time
import datetime
import copy
#
# layers format:
# [ ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.bn1", (256, 480, 512)],
#   ...
# ]



# [1,22,49,54], 3, [100,210,210,470,470]
def mobilenet_v2_space(model,dna, hw_cconv, hw_dconv, args):
    global p3size

    pattern_idx, pattern_do_or_not, q_list, hw_port = dna[0:4], dna[4:6], dna[6:24], dna[24:26]

    hw_cconv[5] += hw_port[0]
    hw_cconv[6] += hw_port[1]
    hw_cconv[7] = 32 - hw_cconv[5] - hw_cconv[6]

    hw_dconv[5], hw_dconv[6], hw_dconv[7] = hw_cconv[5], hw_cconv[6], hw_cconv[7]


    pattern_space = pattern_sets_generate_3((3, 3),p3size)

    pattern = {}
    i = 0
    for idx in pattern_idx:
        pattern[i] = pattern_space[idx].reshape((3, 3))
        i+=1

    layer_names = ["features.0.0","features.1.conv.0.0"]
    select_layer = []
    for i in range(2):
        if pattern_do_or_not[i]==1:
            select_layer.append(layer_names[i])





    quan_paras = {}

    quan_paras["features.7.conv.2"] = [1, q_list[0], True]
    quan_paras["features.8.conv.2"] = [1, q_list[1], True]
    quan_paras["features.9.conv.2"] = [1, q_list[2], True]
    quan_paras["features.10.conv.2"] = [1, q_list[3], True]
    quan_paras["features.11.conv.2"] = [1, q_list[4], True]
    quan_paras["features.12.conv.0.0"] = [1, q_list[5], True]
    quan_paras["features.12.conv.2"] = [1, q_list[6], True]
    quan_paras["features.13.conv.0.0"] = [1, q_list[7], True]
    quan_paras["features.13.conv.2"] = [1, q_list[8], True]
    quan_paras["features.14.conv.0.0"] = [1, q_list[9], True]
    quan_paras["features.14.conv.2"] = [1, q_list[10], True]
    quan_paras["features.15.conv.0.0"] = [1, q_list[11], True]
    quan_paras["features.15.conv.2"] = [1, q_list[12], True]
    quan_paras["features.16.conv.0.0"] = [1, q_list[13], True]
    quan_paras["features.16.conv.2"] = [1, q_list[14], True]
    quan_paras["features.17.conv.0.0"] = [1, q_list[15], True]
    quan_paras["features.17.conv.2"] = [1, q_list[16], True]
    quan_paras["features.18.0"] = [1, q_list[17], True]

    model_modify.Kernel_Patter(model, select_layer, pattern, args)
    model_modify.Kenel_Quantization(model, quan_paras.keys(), quan_paras)

    return model, hw_cconv, hw_dconv


def get_space():
    global p3size
    p3size = 3
    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    p3num = len(pattern_33_space.keys())
    print(p3num)
    space_name = ("KP", "KP", "KP", "KP",
                  "sKP", "sKP",
                  'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu',
                  "HW","HW")

    space = (list(range(p3num)), list(range(p3num)), list(range(p3num)), list(range(p3num)),
             [0,1],[0,1],
             [14, 15], [14, 15], [14, 15], [14, 15], [14, 15], [14, 15], [14, 15], [14, 15], [14, 15], [14, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15],
             [-3,-2, -1, 0, 1, 2,3], [-1, 0, 1, 2,3, 4, 5])

    return space_name,space


def dna_analysis(dna, logger):
    global p3size

    pat_point, pattern_do_or_not, quant_point, comm_point = dna[0:4], dna[4:6], dna[6:24], dna[24:26]

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)

    for p in pat_point:
        logger.info("--------->Pattern 3-3 {}: {}".format(p, pattern_33_space[p].flatten()))
    logger.info("--------->Weight Pruning or Not: {}".format(pattern_do_or_not))
    logger.info("--------->Qunatization: {}".format(quant_point))
    logger.info("--------->HW: {}".format(comm_point))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser('Parser User Input Arguments')
    # parser.add_argument('--device', default='cpu', help='device')
    # args = parser.parse_args()

    args = train.parse_args()
    # data_loader,data_loader_test = train.get_data_loader(args)


    model_name = "mobilenet_v2"
    dataset_name = "cifar10"

    hw_dconv_str = "960, 1, 32, 32, 3, 14, 6, 10"
    hw_cconv_str = "160, 9, 32, 32, 3, 14, 6, 10"
    oriDHW = [int(x.strip()) for x in hw_dconv_str.split(",")]
    oriCHW = [int(x.strip()) for x in hw_cconv_str.split(",")]

    start_time = time.time()
    count = 60
    record = {}
    latency = []
    for i in range(count):

        model = getattr(cifar10_models, model_name)(pretrained=True)


        # model = globals()["resnet18"]()

        _, space = get_space()
        dna = []
        for selection in space:
            dna.append(random.choice(selection))
        print(dna)

        DHW = copy.deepcopy(oriDHW)
        CHW = copy.deepcopy(oriCHW)

        model,DHW,CHW = mobilenet_v2_space(model, dna, DHW, CHW, args)

        model = model.to(args.device)

        print("=" * 10, model_name, "performance analysis:")
        total_lat = bottlenect_conv_dconv.get_performance(model, dataset_name, DHW, CHW, args.device)
        print(total_lat)
        latency.append(total_lat)

        # acc1,acc5,_ = train.main(args, dna, HW, data_loader, data_loader_test)
        # record[i] = (acc5,total_lat)
        # print("Random {}: acc-{}, lat-{}".format(i, acc5,total_lat))
        # print(dna)
        # print("=" * 100)

    print("="*100)

    print("Min latency:",min(latency),max(latency),sum(latency)/float(len(latency)))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # print("Exploration End, using time {}".format(total_time_str))
    # for k,v in record.items():
    #     print(k,v)
    # print(min(latency), max(latency), sum(latency) / len(latency))


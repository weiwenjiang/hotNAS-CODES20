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

    pattern_idx, pattern_do_or_not, q_list, hw_port = dna[0:4], dna[4:6], dna[6:14], dna[14:16]

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

    layer_names = ['features.0.0', 'features.1.conv.0.0']
    select_layer = []
    for i in range(2):
        if pattern_do_or_not[i]==1:
            select_layer.append(layer_names[i])





    quan_paras = {}

    quan_paras["features.14.conv.2"] = [1, q_list[0], True]
    quan_paras["features.15.conv.0.0"] = [1, q_list[1], True]
    quan_paras["features.15.conv.2"] = [1, q_list[2], True]
    quan_paras["features.16.conv.0.0"] = [1, q_list[3], True]
    quan_paras["features.16.conv.2"] = [1, q_list[4], True]
    quan_paras["features.17.conv.0.0"] = [1, q_list[5], True]
    quan_paras["features.17.conv.2"] = [1, q_list[6], True]
    quan_paras["features.18.0"] = [1, q_list[7], True]

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
                  'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu',
                  "HW","HW")

    space = (list(range(p3num)), list(range(p3num)), list(range(p3num)), list(range(p3num)),
             [0,1],[0,1],
             [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15],
             [3, 6, 9, 12, 15], [3, 6, 9, 12, 15], [3, 6, 9, 12, 15],
             [-5,-4,-3,-2, -1, 0, 1, 2,3], [-5,-4,-3,-2,-1, 0, 1, 2,3])

    return space_name,space


def dna_analysis(dna, logger):
    global p3size

    pat_point, pattern_do_or_not, quant_point, comm_point = dna[0:4], dna[4:6], dna[6:14], dna[14:16]

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
    dataset_name = "imagenet"

    hw_dconv_str = "576, 1, 32, 32, 3, 10, 10, 10"
    hw_cconv_str = "160, 12, 32, 32, 3, 10, 10, 10"
    oriDHW = [int(x.strip()) for x in hw_dconv_str.split(",")]
    oriCHW = [int(x.strip()) for x in hw_cconv_str.split(",")]




    start_time = time.time()
    count = 60
    record = {}
    latency = []
    for i in range(count):

        model = globals()[model_name](pretrained=args.pretrained)


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


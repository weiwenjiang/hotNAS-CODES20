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
def resnet_18_space(model,dna, hw_cconv, args):
    global p3size

    pattern_idx, pattern_do_or_not, q_list, hw_port = dna[0:4], dna[4:9], dna[9:23], dna[23:25]

    hw_cconv[5] += hw_port[0]
    hw_cconv[6] += hw_port[1]
    hw_cconv[7] = 32 - hw_cconv[5] - hw_cconv[6]

    # print(hw_cconv)

    pattern_space = pattern_sets_generate_3((3, 3),p3size)

    pattern = {}
    i = 0
    for idx in pattern_idx:
        pattern[i] = pattern_space[idx].reshape((3, 3))
        i+=1

    layer_names = ["conv1","layer1.0.conv1","layer1.0.conv2",
                   "layer1.1.conv1","layer1.1.conv2"]
    select_layer = []
    for i in range(5):
        if pattern_do_or_not[i]==1:
            select_layer.append(layer_names[i])




    quan_paras = {}

    quan_paras["layer2.0.conv1"] = [1, q_list[0], True]
    quan_paras["layer2.0.conv2"] = [1, q_list[1], True]
    quan_paras["layer2.1.conv1"] = [1, q_list[2], True]
    quan_paras["layer2.1.conv2"] = [1, q_list[3], True]
    quan_paras["layer3.0.conv1"] = [1, q_list[4], True]
    quan_paras["layer3.0.conv2"] = [1, q_list[5], True]
    quan_paras["layer3.0.downsample.0"] = [1, q_list[6], True]
    quan_paras["layer3.1.conv1"] = [1, q_list[7], True]
    quan_paras["layer3.1.conv2"] = [1, q_list[8], True]
    quan_paras["layer4.0.conv1"] = [1, q_list[9], True]
    quan_paras["layer4.0.conv2"] = [1, q_list[10], True]
    quan_paras["layer4.0.downsample.0"] = [1, q_list[11], True]
    quan_paras["layer4.1.conv1"] = [1, q_list[12], True]
    quan_paras["layer4.1.conv2"] = [1, q_list[13], True]


    model_modify.Kernel_Patter(model, select_layer, pattern, args)
    model_modify.Kenel_Quantization(model, quan_paras.keys(), quan_paras)

    return model,hw_cconv


def get_space():
    global p3size
    p3size = 3
    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    p3num = len(pattern_33_space.keys())

    space_name = ("KP", "KP", "KP", "KP",
                  "sKP", "sKP", "sKP", "sKP", "sKP",
                  'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu',
                  "HW","HW")

    space = (list(range(p3num)), list(range(p3num)), list(range(p3num)), list(range(p3num)),
             [0,1],[0,1],[0,1],[0,1],[0,1],
             [7, 9, 11, 13, 15], [11, 12, 13, 14, 15, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16], [4, 7, 10, 13, 16],
             [-3,-2, -1, 0, 1, 2,3], [-1, 0, 1, 2,3, 4, 5])

    return space_name,space


def dna_analysis(dna, logger):
    global p3size

    pat_point, pattern_do_or_not, quant_point, comm_point = dna[0:4], dna[4:9], dna[9:23], dna[23:25]

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


    model_name = "resnet18"
    dataset_name = "cifar10"

    hw_str = "130, 19, 32, 32, 3, 18, 2, 10"
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in hw_str.split(",")]
    oriHW = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
    start_time = time.time()
    count = 60
    record = {}
    for i in range(count):

        model = getattr(cifar10_models, model_name)(pretrained=True)


        # model = globals()["resnet18"]()

        _, space = get_space()
        dna = []
        for selection in space:
            dna.append(random.choice(selection))
        print(dna)

        HW = copy.deepcopy(oriHW)

        model,HW = resnet_18_space(model, dna, HW, args)

        model = model.to(args.device)

        print("=" * 10, model_name, "performance analysis:")
        total_lat = bottlenect_conv_dconv.get_performance(model, dataset_name, HW, HW, args.device)
        print(total_lat)
        # latency.append(total_lat)

        # acc1,acc5,_ = train.main(args, dna, HW, data_loader, data_loader_test)
        # record[i] = (acc5,total_lat)
        # print("Random {}: acc-{}, lat-{}".format(i, acc5,total_lat))
        # print(dna)
        # print("=" * 100)

    print("="*100)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # print("Exploration End, using time {}".format(total_time_str))
    # for k,v in record.items():
    #     print(k,v)
    # print(min(latency), max(latency), sum(latency) / len(latency))


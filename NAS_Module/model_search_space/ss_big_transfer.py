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
def big_transfer_space(model,dna, hw_cconv, args):
    global p3size

    q_list, hw_port = dna[0:46], dna[46:48]

    hw_cconv[5] += hw_port[0]
    hw_cconv[6] += hw_port[1]
    hw_cconv[7] = 32 - hw_cconv[5] - hw_cconv[6]

    print(hw_cconv)

    # pattern_space = pattern_sets_generate_3((3, 3),p3size)
    #
    # pattern = {}
    # i = 0
    # for idx in pattern_idx:
    #     pattern[i] = pattern_space[idx].reshape((3, 3))
    #     i+=1
    #
    # layer_names = ["conv1","layer1.0.conv1","layer1.0.conv2",
    #                "layer1.1.conv1","layer1.1.conv2"]
    # select_layer = []
    # for i in range(5):
    #     if pattern_do_or_not[i]==1:
    #         select_layer.append(layer_names[i])
    #
    #


    quan_paras = {}

    quan_paras["module.root.conv"] = [2, q_list[0], True]
    quan_paras["module.body.block1.unit01.conv2"] = [2, q_list[1], True]
    quan_paras["module.body.block1.unit02.conv2"] = [2, q_list[2], True]
    quan_paras["module.body.block1.unit03.conv2"] = [2, q_list[3], True]
    quan_paras["module.body.block2.unit01.conv2"] = [2, q_list[4], True]
    quan_paras["module.body.block2.unit01.conv3"] = [3, q_list[5], True]
    quan_paras["module.body.block2.unit01.downsample"] = [2, q_list[6], True]
    quan_paras["module.body.block2.unit02.conv1"] = [2, q_list[7], True]
    quan_paras["module.body.block2.unit02.conv2"] = [2, q_list[8], True]
    quan_paras["module.body.block2.unit02.conv3"] = [3, q_list[9], True]
    quan_paras["module.body.block2.unit03.conv1"] = [2, q_list[10], True]
    quan_paras["module.body.block2.unit03.conv2"] = [2, q_list[11], True]
    quan_paras["module.body.block2.unit03.conv3"] = [3, q_list[12], True]
    quan_paras["module.body.block2.unit04.conv1"] = [2, q_list[13], True]
    quan_paras["module.body.block2.unit04.conv2"] = [2, q_list[14], True]
    quan_paras["module.body.block2.unit04.conv3"] = [3, q_list[15], True]
    quan_paras["module.body.block3.unit01.conv1"] = [2, q_list[16], True]
    quan_paras["module.body.block3.unit01.conv2"] = [2, q_list[17], True]
    quan_paras["module.body.block3.unit01.conv3"] = [3, q_list[18], True]
    quan_paras["module.body.block3.unit01.downsample"] = [2, q_list[19], True]
    quan_paras["module.body.block3.unit02.conv1"] = [2, q_list[20], True]
    quan_paras["module.body.block3.unit02.conv2"] = [2, q_list[21], True]
    quan_paras["module.body.block3.unit02.conv3"] = [3, q_list[22], True]
    quan_paras["module.body.block3.unit03.conv1"] = [2, q_list[23], True]
    quan_paras["module.body.block3.unit03.conv2"] = [2, q_list[24], True]
    quan_paras["module.body.block3.unit03.conv3"] = [3, q_list[25], True]
    quan_paras["module.body.block3.unit04.conv1"] = [2, q_list[26], True]
    quan_paras["module.body.block3.unit04.conv2"] = [2, q_list[27], True]
    quan_paras["module.body.block3.unit04.conv3"] = [3, q_list[28], True]
    quan_paras["module.body.block3.unit05.conv1"] = [2, q_list[29], True]
    quan_paras["module.body.block3.unit05.conv2"] = [2, q_list[30], True]
    quan_paras["module.body.block3.unit05.conv3"] = [2, q_list[31], True]
    quan_paras["module.body.block3.unit06.conv1"] = [2, q_list[32], True]
    quan_paras["module.body.block3.unit06.conv2"] = [2, q_list[33], True]
    quan_paras["module.body.block3.unit06.conv3"] = [3, q_list[34], True]
    quan_paras["module.body.block4.unit01.conv1"] = [2, q_list[35], True]
    quan_paras["module.body.block4.unit01.conv2"] = [2, q_list[36], True]
    quan_paras["module.body.block4.unit01.conv3"] = [2, q_list[37], True]
    quan_paras["module.body.block4.unit01.downsample"] = [2, q_list[38], True]
    quan_paras["module.body.block4.unit02.conv1"] = [2, q_list[39], True]
    quan_paras["module.body.block4.unit02.conv2"] = [2, q_list[40], True]
    quan_paras["module.body.block4.unit02.conv3"] = [2, q_list[41], True]
    quan_paras["module.body.block4.unit03.conv1"] = [2, q_list[42], True]
    quan_paras["module.body.block4.unit03.conv2"] = [2, q_list[43], True]
    quan_paras["module.body.block4.unit03.conv3"] = [2, q_list[44], True]
    quan_paras["module.head.conv"] = [2, q_list[45], True]



    # model_modify.Kernel_Patter(model, select_layer, pattern, args)
    model_modify.Kenel_Quantization(model, quan_paras.keys(), quan_paras, is_std_conv=True)

    return model,hw_cconv


def get_space():
    global p3size
    p3size = 3
    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    p3num = len(pattern_33_space.keys())
    print(p3num)
    space_name = ('Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu',
                  "HW","HW")

    space = ([14], [9, 10, 11, 12, 13, 14], [9, 10, 11, 12, 13, 14], [9, 10, 11, 12, 13, 14], [4, 6, 8, 10, 12, 14],
     [5, 7, 9, 11, 13], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12],
     [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14],
     [4, 6, 8, 10, 12], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12, 14],
     [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14],
     [4, 6, 8, 10, 12], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12], [4, 6, 8, 10, 12, 14],
     [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12],
     [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14],
     [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14],
     [4, 6, 8, 10, 12, 14], [-3,-2, -1, 0, 1, 2, 3], [-3,-2, -1, 0, 1, 2, 3],)

    return space_name,space


def dna_analysis(dna, logger):
    global p3size

    quant_point, comm_point = dna[0:46], dna[46:48]

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)

    # for p in pat_point:
    #     logger.info("--------->Pattern 3-3 {}: {}".format(p, pattern_33_space[p].flatten()))
    # logger.info("--------->Weight Pruning or Not: {}".format(pattern_do_or_not))
    logger.info("--------->Qunatization: {}".format(quant_point))
    logger.info("--------->HW: {}".format(comm_point))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser('Parser User Input Arguments')
    # parser.add_argument('--device', default='cpu', help='device')
    # args = parser.parse_args()

    args = train.parse_args()
    # data_loader,data_loader_test = train.get_data_loader(args)


    model_name = "big_transfer"
    dataset_name = "cifar10"

    hw_str = "130, 19, 32, 32, 7, 18, 6, 6"
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in hw_str.split(",")]
    oriHW = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
    start_time = time.time()
    count = 60
    record = {}
    latency = []
    for i in range(count):

        model = getattr(cifar10_models, model_name)(pretrained=False)


        _, space = get_space()
        dna = []
        for selection in space:
            dna.append(random.choice(selection))
        print(dna)

        HW = copy.deepcopy(oriHW)

        model,HW = big_transfer_space(model, dna, HW, args)

        model = model.to(args.device)

        print("=" * 10, model_name, "performance analysis:")
        total_lat = bottlenect_conv_dconv.get_performance(model, dataset_name, HW, HW, args.device)
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


# 3.5460575008884154 6.728768956582633 4.804184316888049 (60 random results)
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
def densenet121_space(model,dna, hw_cconv, args):
    global p3size

    pattern_idx, pattern_do_or_not, q_list, hw_port = dna[0:4], dna[4:10], dna[10:103], dna[103:105]

    hw_cconv[5] += hw_port[0]
    hw_cconv[6] += hw_port[1]
    hw_cconv[7] = 32 - hw_cconv[5] - hw_cconv[6]

    print(hw_cconv)

    pattern_space = pattern_sets_generate_3((3, 3),p3size)

    pattern = {}
    i = 0
    for idx in pattern_idx:
        pattern[i] = pattern_space[idx].reshape((3, 3))
        i+=1

    layer_names = ['features.denseblock1.denselayer1.conv2', 'features.denseblock1.denselayer2.conv2', 'features.denseblock1.denselayer3.conv2', 'features.denseblock1.denselayer4.conv2', 'features.denseblock1.denselayer5.conv2', 'features.denseblock1.denselayer6.conv2']
    select_layer = []
    for i in range(5):
        if pattern_do_or_not[i]==1:
            select_layer.append(layer_names[i])




    quan_paras = {}

    quan_paras["features.denseblock2.denselayer1.conv2"] = [1, q_list[0], True]
    quan_paras["features.denseblock2.denselayer2.conv2"] = [1, q_list[1], True]
    quan_paras["features.denseblock2.denselayer3.conv2"] = [1, q_list[2], True]
    quan_paras["features.denseblock2.denselayer4.conv2"] = [1, q_list[3], True]
    quan_paras["features.denseblock2.denselayer5.conv2"] = [1, q_list[4], True]
    quan_paras["features.denseblock2.denselayer6.conv2"] = [1, q_list[5], True]
    quan_paras["features.denseblock2.denselayer7.conv2"] = [1, q_list[6], True]
    quan_paras["features.denseblock2.denselayer8.conv2"] = [1, q_list[7], True]
    quan_paras["features.denseblock2.denselayer9.conv2"] = [1, q_list[8], True]
    quan_paras["features.denseblock2.denselayer10.conv2"] = [1, q_list[9], True]
    quan_paras["features.denseblock2.denselayer11.conv2"] = [1, q_list[10], True]
    quan_paras["features.denseblock2.denselayer12.conv2"] = [1, q_list[11], True]
    quan_paras["features.denseblock3.denselayer1.conv1"] = [1, q_list[12], True]
    quan_paras["features.denseblock3.denselayer1.conv2"] = [1, q_list[13], True]
    quan_paras["features.denseblock3.denselayer2.conv1"] = [1, q_list[14], True]
    quan_paras["features.denseblock3.denselayer2.conv2"] = [1, q_list[15], True]
    quan_paras["features.denseblock3.denselayer3.conv1"] = [1, q_list[16], True]
    quan_paras["features.denseblock3.denselayer3.conv2"] = [1, q_list[17], True]
    quan_paras["features.denseblock3.denselayer4.conv1"] = [1, q_list[18], True]
    quan_paras["features.denseblock3.denselayer4.conv2"] = [1, q_list[19], True]
    quan_paras["features.denseblock3.denselayer5.conv1"] = [1, q_list[20], True]
    quan_paras["features.denseblock3.denselayer5.conv2"] = [1, q_list[21], True]
    quan_paras["features.denseblock3.denselayer6.conv1"] = [1, q_list[22], True]
    quan_paras["features.denseblock3.denselayer6.conv2"] = [1, q_list[23], True]
    quan_paras["features.denseblock3.denselayer7.conv1"] = [1, q_list[24], True]
    quan_paras["features.denseblock3.denselayer7.conv2"] = [1, q_list[25], True]
    quan_paras["features.denseblock3.denselayer8.conv1"] = [1, q_list[26], True]
    quan_paras["features.denseblock3.denselayer8.conv2"] = [1, q_list[27], True]
    quan_paras["features.denseblock3.denselayer9.conv1"] = [1, q_list[28], True]
    quan_paras["features.denseblock3.denselayer9.conv2"] = [1, q_list[29], True]
    quan_paras["features.denseblock3.denselayer10.conv1"] = [1, q_list[30], True]
    quan_paras["features.denseblock3.denselayer10.conv2"] = [1, q_list[31], True]
    quan_paras["features.denseblock3.denselayer11.conv1"] = [1, q_list[32], True]
    quan_paras["features.denseblock3.denselayer11.conv2"] = [1, q_list[33], True]
    quan_paras["features.denseblock3.denselayer12.conv1"] = [1, q_list[34], True]
    quan_paras["features.denseblock3.denselayer12.conv2"] = [1, q_list[35], True]
    quan_paras["features.denseblock3.denselayer13.conv1"] = [1, q_list[36], True]
    quan_paras["features.denseblock3.denselayer13.conv2"] = [1, q_list[37], True]
    quan_paras["features.denseblock3.denselayer14.conv1"] = [1, q_list[38], True]
    quan_paras["features.denseblock3.denselayer14.conv2"] = [1, q_list[39], True]
    quan_paras["features.denseblock3.denselayer15.conv1"] = [1, q_list[40], True]
    quan_paras["features.denseblock3.denselayer15.conv2"] = [1, q_list[41], True]
    quan_paras["features.denseblock3.denselayer16.conv1"] = [1, q_list[42], True]
    quan_paras["features.denseblock3.denselayer16.conv2"] = [1, q_list[43], True]
    quan_paras["features.denseblock3.denselayer17.conv1"] = [1, q_list[44], True]
    quan_paras["features.denseblock3.denselayer17.conv2"] = [1, q_list[45], True]
    quan_paras["features.denseblock3.denselayer18.conv1"] = [1, q_list[46], True]
    quan_paras["features.denseblock3.denselayer18.conv2"] = [1, q_list[47], True]
    quan_paras["features.denseblock3.denselayer19.conv1"] = [1, q_list[48], True]
    quan_paras["features.denseblock3.denselayer19.conv2"] = [1, q_list[49], True]
    quan_paras["features.denseblock3.denselayer20.conv1"] = [1, q_list[50], True]
    quan_paras["features.denseblock3.denselayer20.conv2"] = [1, q_list[51], True]
    quan_paras["features.denseblock3.denselayer21.conv1"] = [1, q_list[52], True]
    quan_paras["features.denseblock3.denselayer21.conv2"] = [1, q_list[53], True]
    quan_paras["features.denseblock3.denselayer22.conv1"] = [1, q_list[54], True]
    quan_paras["features.denseblock3.denselayer22.conv2"] = [1, q_list[55], True]
    quan_paras["features.denseblock3.denselayer23.conv1"] = [1, q_list[56], True]
    quan_paras["features.denseblock3.denselayer23.conv2"] = [1, q_list[57], True]
    quan_paras["features.denseblock3.denselayer24.conv1"] = [1, q_list[58], True]
    quan_paras["features.denseblock3.denselayer24.conv2"] = [1, q_list[59], True]
    quan_paras["features.transition3.conv"] = [1, q_list[60], True]
    quan_paras["features.denseblock4.denselayer1.conv1"] = [1, q_list[61], True]
    quan_paras["features.denseblock4.denselayer1.conv2"] = [1, q_list[62], True]
    quan_paras["features.denseblock4.denselayer2.conv1"] = [1, q_list[63], True]
    quan_paras["features.denseblock4.denselayer2.conv2"] = [1, q_list[64], True]
    quan_paras["features.denseblock4.denselayer3.conv1"] = [1, q_list[65], True]
    quan_paras["features.denseblock4.denselayer3.conv2"] = [1, q_list[66], True]
    quan_paras["features.denseblock4.denselayer4.conv1"] = [1, q_list[67], True]
    quan_paras["features.denseblock4.denselayer4.conv2"] = [1, q_list[68], True]
    quan_paras["features.denseblock4.denselayer5.conv1"] = [1, q_list[69], True]
    quan_paras["features.denseblock4.denselayer5.conv2"] = [1, q_list[70], True]
    quan_paras["features.denseblock4.denselayer6.conv1"] = [1, q_list[71], True]
    quan_paras["features.denseblock4.denselayer6.conv2"] = [1, q_list[72], True]
    quan_paras["features.denseblock4.denselayer7.conv1"] = [1, q_list[73], True]
    quan_paras["features.denseblock4.denselayer7.conv2"] = [1, q_list[74], True]
    quan_paras["features.denseblock4.denselayer8.conv1"] = [1, q_list[75], True]
    quan_paras["features.denseblock4.denselayer8.conv2"] = [1, q_list[76], True]
    quan_paras["features.denseblock4.denselayer9.conv1"] = [1, q_list[77], True]
    quan_paras["features.denseblock4.denselayer9.conv2"] = [1, q_list[78], True]
    quan_paras["features.denseblock4.denselayer10.conv1"] = [1, q_list[79], True]
    quan_paras["features.denseblock4.denselayer10.conv2"] = [1, q_list[80], True]
    quan_paras["features.denseblock4.denselayer11.conv1"] = [1, q_list[81], True]
    quan_paras["features.denseblock4.denselayer11.conv2"] = [1, q_list[82], True]
    quan_paras["features.denseblock4.denselayer12.conv1"] = [1, q_list[83], True]
    quan_paras["features.denseblock4.denselayer12.conv2"] = [1, q_list[84], True]
    quan_paras["features.denseblock4.denselayer13.conv1"] = [1, q_list[85], True]
    quan_paras["features.denseblock4.denselayer13.conv2"] = [1, q_list[86], True]
    quan_paras["features.denseblock4.denselayer14.conv1"] = [1, q_list[87], True]
    quan_paras["features.denseblock4.denselayer14.conv2"] = [1, q_list[88], True]
    quan_paras["features.denseblock4.denselayer15.conv1"] = [1, q_list[89], True]
    quan_paras["features.denseblock4.denselayer15.conv2"] = [1, q_list[90], True]
    quan_paras["features.denseblock4.denselayer16.conv1"] = [1, q_list[91], True]
    quan_paras["features.denseblock4.denselayer16.conv2"] = [1, q_list[92], True]

    model_modify.Kernel_Patter(model, select_layer, pattern, args)
    model_modify.Kenel_Quantization(model, quan_paras.keys(), quan_paras)

    return model,hw_cconv


def get_space():
    global p3size
    p3size = 3
    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)
    p3num = len(pattern_33_space.keys())
    print(p3num)
    space_name = ("KP", "KP", "KP", "KP",
                  "sKP", "sKP", "sKP", "sKP", "sKP", "sKP",
                  'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu', 'Qu',
                  "HW","HW")

    space = (list(range(p3num)), list(range(p3num)), list(range(p3num)), list(range(p3num)),
             [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
             [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [4, 6, 8, 10, 12, 14], [10, 11, 12, 13, 14, 15], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [1, 4, 7, 10, 13],
             [-3,-2, -1, 0, 1, 2,3], [-3,-2, -1, 0, 1, 2,3])

    return space_name,space


def dna_analysis(dna, logger):
    global p3size

    pat_point, pattern_do_or_not, quant_point, comm_point = dna[0:4], dna[4:10], dna[10:103], dna[103:105]

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


    model_name = "densenet121"
    dataset_name = "cifar10"

    hw_str = "70, 36, 32, 32, 3, 18, 6, 6"
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in hw_str.split(",")]
    oriHW = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
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

        HW = copy.deepcopy(oriHW)

        model,HW = densenet121_space(model, dna, HW, args)

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


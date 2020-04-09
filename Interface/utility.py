import sys
import math

def is_same(shape):
    k_size = shape
    if len(shape) == 2 and shape[0] != shape[1]:
        print("Not support kernel/stride/padding with different size")
        sys.exit(0)
    elif len(shape) == 2:
        k_size = shape[0]
    return k_size


def re_quantize(x, total_num = 16, signed=True):
    if signed:
        int_num = 1
    else:
        int_num = 0

    y = math.ceil(x)

    while y!=1:
        y = math.ceil(y/2)
        int_num+=1
        if int_num > total_num:
            return total_num,0

    return int_num,total_num-int_num

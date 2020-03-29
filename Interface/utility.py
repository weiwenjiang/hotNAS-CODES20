import sys

def is_same(shape):
    k_size = shape
    if len(shape) == 2 and shape[0] != shape[1]:
        print("Not support kernel/stride/padding with different size")
        sys.exit(0)
    elif len(shape) == 2:
        k_size = shape[0]
    return k_size
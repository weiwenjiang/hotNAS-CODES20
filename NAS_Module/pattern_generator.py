import torch


# A utility to help in generate a set of patterns

def pattern_sets_generate_3(kernal_shape, zero_num=3):
    num_one = kernal_shape[0] * kernal_shape[1]
    base_tensor = torch.ones(num_one)

    pattern_space = {}

    if kernal_shape[0]==3 and kernal_shape[1]==3:

        if zero_num==3:
            pattern_idx = 0
            for i in range(num_one-2):
                for j in range(i+1,num_one-1):
                    for k in range(j+1,num_one):
                        if i==4 or j==4 or k==4:
                            continue
                        tmp_tensor = base_tensor.clone()
                        tmp_tensor[i] = 0
                        tmp_tensor[j] = 0
                        tmp_tensor[k] = 0
                        pattern_space[pattern_idx] = tmp_tensor
                        pattern_idx+=1
        elif zero_num==2:
            pattern_idx = 0
            for i in range(num_one - 2):
                for j in range(i + 1, num_one - 1):
                    if i == 4 or j == 4:
                        continue
                    tmp_tensor = base_tensor.clone()
                    tmp_tensor[i] = 0
                    tmp_tensor[j] = 0
                    pattern_space[pattern_idx] = tmp_tensor
                    pattern_idx += 1

        elif zero_num==1:
            pattern_idx = 0
            for i in range(num_one - 2):
                    if i == 4:
                        continue
                    tmp_tensor = base_tensor.clone()
                    tmp_tensor[i] = 0
                    pattern_space[pattern_idx] = tmp_tensor
                    pattern_idx += 1

    elif kernal_shape[0] == 5 and kernal_shape[1] == 5:
        fix_one_set = [2,10,14,22,6,7,8,11,12,13,16,17,18]

        if zero_num==6:
            pattern_idx = 0
            for i in range(num_one - 5):
                for j in range(i + 1, num_one - 4):
                    for k in range(j + 1, num_one - 3):
                        for l in range(k + 1, num_one - 2):
                            for m in range(l + 1, num_one - 1):
                                for n in range(m + 1, num_one - 0):
                                    if i in fix_one_set or j in fix_one_set or k in fix_one_set or l in fix_one_set\
                                            or m in fix_one_set or n in fix_one_set:
                                        continue
                                    tmp_tensor = base_tensor.clone()
                                    tmp_tensor[i] = 0
                                    tmp_tensor[j] = 0
                                    tmp_tensor[k] = 0
                                    tmp_tensor[l] = 0
                                    tmp_tensor[m] = 0
                                    tmp_tensor[n] = 0
                                    pattern_space[pattern_idx] = tmp_tensor
                                    pattern_idx += 1
        elif zero_num==8:
            fix_one_set = [2, 10, 14, 22, 6, 7, 8, 11, 12, 13, 16, 17, 18]
            pattern_idx = 0
            for i in range(num_one - 5):
                for j in range(i + 1, num_one - 4):
                    for k in range(j + 1, num_one - 3):
                        for l in range(k + 1, num_one - 2):
                            for m in range(l + 1, num_one - 1):
                                for n in range(m + 1, num_one - 0):
                                    for p in range(n + 1, num_one - 1):
                                        for q in range(p + 1, num_one - 0):

                                            if i in fix_one_set or j in fix_one_set or k in fix_one_set or l in fix_one_set \
                                                    or m in fix_one_set or n in fix_one_set or p in fix_one_set \
                                                    or q in fix_one_set:
                                                continue
                                            tmp_tensor = base_tensor.clone()
                                            tmp_tensor[i] = 0
                                            tmp_tensor[j] = 0
                                            tmp_tensor[k] = 0
                                            tmp_tensor[l] = 0
                                            tmp_tensor[m] = 0
                                            tmp_tensor[n] = 0
                                            tmp_tensor[p] = 0
                                            tmp_tensor[q] = 0
                                            pattern_space[pattern_idx] = tmp_tensor
                                            pattern_idx += 1

    elif kernal_shape[0]==7 and kernal_shape[1]==7:

        pattern_space[0] = torch.tensor([[0., 0., 1., 1., 1., 0., 0.],
                                         [0., 0., 1., 1., 1., 0., 0.],
                                         [0., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 0., 1., 1., 1., 0., 0.],
                                         [0., 0., 0., 1., 0., 0., 0.]]).reshape(7*7)

        pattern_space[1] = torch.tensor([[1., 0., 0., 0., 0., 0., 1.],
                                         [0., 0., 1., 1., 1., 0., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 0., 1., 1., 1., 0., 0.],
                                         [1., 0., 0., 0., 0., 0., 1.]]).reshape(7 * 7)

        pattern_space[2] = torch.tensor([[0., 0., 0., 0., 0., 0., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 1., 1., 1., 1., 1., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.]]).reshape(7 * 7)

        pattern_space[3] = torch.tensor([[1., 1., 1., 0., 1., 1., 1.],
                                         [1., 0., 0., 0., 0., 0., 1.],
                                         [1., 0., 1., 1., 1., 0., 1.],
                                         [0., 0., 1., 1., 1., 0., 0.],
                                         [1., 0., 1., 1., 1., 0., 1.],
                                         [1., 0., 0., 0., 0., 0., 1.],
                                         [1., 1., 0., 0., 0., 1., 1.]]).reshape(7 * 7)

    # for k,v in pattern_space.items():
    #     print(k,v)
    # print(base_tensor)
    return pattern_space



import random
import sys

if __name__ == "__main__":





    pattern_space = pattern_sets_generate_3((5,5))

    # for k,v in pattern_space.items():
    #     print(k,v.reshape((5,5)))
    #     break
    print(len(pattern_space.keys()))
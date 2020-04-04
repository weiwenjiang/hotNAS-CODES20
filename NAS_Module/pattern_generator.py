import torch


# A utility to help in generate a set of patterns

def pattern_sets_generate_3(kernal_shape):
    num_one = kernal_shape[0] * kernal_shape[1]
    base_tensor = torch.ones(num_one)

    pattern_space = {}

    if kernal_shape[0]==3 and kernal_shape[1]==3:
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


if __name__ == "__main__":
    pattern_space = pattern_sets_generate_3((7,7))

    for k,v in pattern_space.items():
        print(k,v.reshape((7,7)))
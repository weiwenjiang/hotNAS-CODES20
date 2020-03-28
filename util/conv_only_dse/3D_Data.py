import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import sys

fh = open("alexnet")
contents = fh.readlines()

dTm = {}
pre_tm = 0
i = 0
for line in contents:
    line = line.strip().replace(" ","").replace("[","").split("]")
    HW1 = [int(x) for x in line[0].split(",")]
    latency = float(line[1])

    if(pre_tm!=HW1[0]):
        pre_tm = HW1[0]
        i+=1
        dTm[i] = []

    dTm[i].append(latency)

    # print(HW1,latency)
    # sys.exit(0)
    # if HW1[0]==64:
    #     i += 1
    #     dTm[i] = []
    #
    # dTm[i].append(latency)





    #
    # if HW1[0] not in dTm.keys():
    #     dTm[HW1[0]] = {}
    # if HW2[0]*1000+HW2[1] not in dTm[HW1[0]].keys():
    #     dTm[HW1[0]][HW2[0] * 1000 + HW2[1]] = []
    #
    # dTm[HW1[0]][HW2[0]*1000+HW2[1]].append(latency)


    # print(HW1,HW2,latency)


for k,v in dTm.items():
    print(k,v)

#
# y = []
#
# for k,v in dTm.items():
#     print(k,v.keys())
#     y = y + list(v.keys())
#
#
# x = list(dTm.keys())
# y = sorted(y)
#
#
#
#
#
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# for k,v in dTm.items():
#     for v_k,v_v in v.items():
#         for z in v_v:
#             ax.scatter3D(k, v_k, z)
#
# #
# # # For each set of style and range settings, plot n random points in the box
# # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
# #     xs = randrange(n, 23, 32)
# #     ys = randrange(n, 0, 100)
# #     zs = randrange(n, zlow, zhigh)
# #     ax.scatter(xs, ys, zs, c=c, marker=m)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
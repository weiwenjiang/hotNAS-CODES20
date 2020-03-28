search_space = {
    'hw_only_cconv': (list(range(10,20,30)),[7,14],[7,14],list(range(2,3,2)),list(range(2,3,2)),list(range(2,3,2))),
    'hw_only_cconv2': (list(range(10,250,30)),[7,14,20,32],[7,14,20,32],list(range(2,10,2)),list(range(2,10,2)),list(range(2,10,2))),
    "cconv_hw": (list(range(8,50,8)),list(range(1,9,1)),[32,64],[32,64],[3],[2],[2],[2]),
    "dconv_hw": (list(range(32,65,8)),list(range(32,65,8)),[5,7,14],[5,7,14],[5],[2],[2],[2])
}


HW_constraints = {
    "r_Ports_BW": 256,
    "r_DSP": 2520,
    "r_BRAM": 1824,
    "r_BRAM_Size": 18000,
    "BITWIDTH": 16,
    "target_HW_Eff": 1000000000
}
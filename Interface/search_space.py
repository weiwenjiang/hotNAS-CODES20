search_space = {
    'hw_only_cconv2': (list(range(10,20,30)),[7,14],[7,14],list(range(2,3,2)),list(range(2,3,2)),list(range(2,3,2))),
    'hw_only_cconv': (list(range(10,250,30)),[7,14,20,32],[7,14,20,32],list(range(2,10,2)),list(range(2,10,2)),list(range(2,10,2))),
    "hw_cd_cconv": (list(range(10,250,30)),[7,14,20,32],[7,14,20,32],list(range(2,10,2)),list(range(2,10,2)),list(range(2,10,2))),
    "hw_cd_dconv": (list(range(10,250,30)),[7,14,20,32],[7,14,20,32],list(range(2,10,2)),list(range(2,10,2)),list(range(2,10,2)))
}


HW_constraints = {
    "r_Ports_BW": 256,
    "r_DSP": 2520,
    "r_BRAM": 1824,
    "r_BRAM_Size": 18000,
    "BITWIDTH": 16,
    "target_HW_Eff": 1000000000
}

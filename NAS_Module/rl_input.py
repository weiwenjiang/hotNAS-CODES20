controller_params = {
    "sw_space": (list(range(56)),list(range(56)),list(range(56)),list(range(56)),list(range(4)),[128],[240,224,208]
                 ,[240,224,208],[496,480,464,448],[496,480,464,448]),
    # dataflow 1, dataflow 2, PE for d1, BW for d1
    "hw_space": (list(range(8,50,8)),list(range(1,9,1)),[32,64],[32,64],[3],[2],[2],[2]),
    'max_episodes': 500,
    "num_children_per_episode": 1,
    "num_hw_per_child": 10,
    'hidden_units': 35,
}


HW_constraints = {
    "r_Ports_BW": 1024,
    "r_DSP": 2520,
    "r_BRAM": 1824,
    "r_BRAM_Size": 18000,
    "BITWIDTH": 16,
    "target_HW_Eff": 1
}
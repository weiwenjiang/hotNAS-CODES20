import PM_Config as config
from math import *
import sys
import PM_Layer


class FPGA_Templates:
    def __init__(self, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p, T, r_PortBW, r_DSP, r_BRAM, r_BRAM_Size=18000, BITWIDTH=16):
        self.Tm = Tm
        self.Tn = Tn
        self.Tr = Tr
        self.Tc = Tc
        self.Tk = Tk
        self.W_p = W_p
        self.I_p = I_p
        self.O_p = O_p
        if T == "cconv" or T == "dconv":
            self.T = T
        else:
            #print("[Error] only support conventional conv2d (cconv) and depthwise conv (dconv)")
            sys.exit(0)
        if T == "dconv":
            if self.Tm != self.Tn:
                self.Tn = self.Tm

        self.usage_DSP = 0
        self.usage_BRAM = 0
        self.usage_comm_bw = 0

        if self.validate_design(Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p, T, r_PortBW, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)==-1:
            # print("[Error] resource cannot satisfied")
            self.Success = False
        else:
            # print("[INOF] construct sucessfully")
            self.Success = True



    def validate_design(self, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p, T, r_PortBW, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH):
        if T == "cconv":
            # Resource validating for conventional conv

            if BITWIDTH==16:
                DSP = Tm*Tn
                Port_BW = 16*(W_p + I_p + O_p)
                if DSP > r_DSP:
                    print("[Error] DSP exceeds")
                    return -1
                if Port_BW > r_PortBW:
                    print("[Error] Communication bitwidth exceeds",W_p,I_p,O_p,Port_BW,r_PortBW)
                    return -1
            elif BITWIDTH==32:
                DSP = 5*Tm*Tn
                Port_BW = 32 * (W_p + I_p + O_p)
                if DSP > r_DSP:
                    #print("[Error] DSP exceeds")
                    return -1
                if Port_BW > r_PortBW:
                    #print("[Error] Communication bitwidth exceeds")
                    return -1
            else:
                #print("[Error] only support 16-bit fixed point and 32-bit floating point")
                sys.exit(0)

            bI = 2*Tn*ceil(Tr*Tc*BITWIDTH/r_BRAM_Size)
            bO = 2*Tm*ceil(Tr*Tc*BITWIDTH/r_BRAM_Size)
            if BITWIDTH == 32:
                bW = 2*Tm*Tn*ceil(Tk*Tk*BITWIDTH/r_BRAM_Size)  # 32-bit floating point
            elif BITWIDTH == 16:
                bW = Tm*Tn*ceil(4*Tk*Tk*BITWIDTH/r_BRAM_Size)  # fix point
            BRAM = bI + bO + bW
            # if BRAM > r_BRAM:
            #     print("[Error] BRAM exceeds - cconv",bI, bO, bW,BRAM,r_BRAM)
            #     return -1

        elif T=="dconv":
            # Resource validating for depthwise conv

            if BITWIDTH == 16:
                DSP = Tm * 1
                Port_BW = 16 * (W_p + I_p + O_p)
                if DSP > r_DSP:
                    print("[Error] DSP exceeds")
                    return -1
                if Port_BW > r_PortBW:
                    print("[Error] Communication bitwidth exceeds")
                    return -1
            elif BITWIDTH == 32:
                DSP = 5 * Tm * 1
                Port_BW = 32 * (W_p + I_p + O_p)
                if DSP > r_DSP:
                    print("[Error] DSP exceeds")
                    return -1
                if Port_BW > r_PortBW:
                    print("[Error] Communication bitwidth exceeds")
                    return -1
            else:
                print("[Error] only support 16-bit fixed point and 32-bit floating point")
                sys.exit(0)

            bI = 2*Tn*ceil(Tr*Tc*BITWIDTH/r_BRAM_Size)
            bO = 2*Tm*ceil(Tr*Tc*BITWIDTH/r_BRAM_Size)
            if BITWIDTH == 32:
                bW = 2*Tm*1*ceil(Tk*Tk*BITWIDTH/r_BRAM_Size)  # 32-bit floating point
            elif BITWIDTH == 16:
                bW = Tm*1*ceil(4*Tk*Tk*BITWIDTH/r_BRAM_Size)  # fix point
            BRAM = bI + bO + bW
            # if BRAM > r_BRAM:
            #     print("[Error] BRAM exceeds - dconv", BRAM, r_BRAM)
            #     return -1

        else:
            #print("[Error] only support conventional conv2d (cconv) and depthwise conv (dconv)")
            sys.exit(0)

        self.usage_DSP = DSP
        self.usage_BRAM = BRAM
        self.usage_comm_bw = Port_BW
        # print("[INFO] Validation successful: DSP {}/{}, BRAM {}/{}, Ports {}/{}".format(DSP,r_DSP,BRAM,r_BRAM,Port_BW,r_PortBW))
        return 1

    def layer_template_match_check(self, Layer):
        [B, M, N, R, C, K, S, T, P] = Layer.getPara()
        [Tm, Tn, Tr, Tc, Tk] = (self.Tm, self.Tn, self.Tr, self.Tc, self.Tk)
        if K > Tk:
            print("[Error] <From class Conv2D_FPGA_Template> Kernel Size {} of input conv is larger than the template supported {}".format(
                   K, Tk))
            return False
        elif K < Tk:
            # print("[Warning] <From class Conv2D_FPGA_Template> {} HW kernel {} is underutilized for conv kernel {}".format(T, Tk,
            #                                                                                                          K))
            pass;

        if self.T != T:
            print("[Error] <From class Conv2D_FPGA_Template> cannot using {} template for {}".format(self.T, T))
            return False
        return True

    def get_cconv_latency(self, Layer, pattern_ones=-1, quan_paras=[]):
        [B, M, N, R, C, K, S, T, P] = Layer.getPara()
        [Tm, Tn, Tr, Tc, Tk] = (self.Tm, self.Tn, self.Tr, self.Tc, self.Tk)
        [W_p, I_p, O_p] = (self.W_p, self.I_p, self.O_p)

        if len(quan_paras)!=0:
            bits = quan_paras[0]+quan_paras[1]
            lat_W_mem = Tm * Tn * K * K / floor(W_p*16/float(bits))
        else:
            lat_W_mem = Tm * Tn * K * K / W_p
        lat_I_mem = min(Tn,N) * min(Tr,R) * min(Tc,C) / I_p
        lat_O_mem = min(Tm,M) * ceil(min(Tr,R) / S) * ceil(min(Tc,C) / S) / O_p
        if pattern_ones!=-1:
            lat_Comp = pattern_ones * ceil(min(Tr, R+P*2) / S) * ceil(min(Tc, C+P*2) / S)
        else:
            lat_Comp = K * K * ceil(min(Tr,R+P*2) / S) * ceil(min(Tc,C+P*2) / S)

        Lat1 = max(lat_I_mem, lat_W_mem, lat_Comp)
        Lat2 = max(ceil(N / Tn) * Lat1, lat_O_mem)
        Lat = B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm) * Lat2  # + (tO_mem + Lat1)

        # print(Lat,(lat_I_mem,lat_W_mem,lat_Comp),(ceil(N/Tn)*Lat1,lat_O_mem))

        if Lat2 == lat_O_mem:
            bottle_neck = "storing OFM"
        elif Lat1 == lat_I_mem:
            bottle_neck = "loading IFM"
        elif Lat1 == lat_W_mem:
            bottle_neck = "loading Weight"
        elif Lat1 == lat_Comp:
            bottle_neck = "computing"
        else:
            bottle_neck = "wired"

        I = lat_I_mem * ceil(N / Tn) * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)
        # print(Tn, min(Tr,R), min(Tc,C), I_p, lat_I_mem, ceil(N / Tn) , B , ceil(R / Tr) , ceil(C / Tc) , ceil(M / Tm), lat_I_mem * ceil(N / Tn) * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm))
        O = lat_O_mem * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)
        W = lat_W_mem * ceil(N / Tn) * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)
        C = lat_Comp  * ceil(N / Tn) * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)

        return Lat, bottle_neck, [I,O,W,C]

    def get_dconv_latency(self, Layer, pattern_ones=-1, quan_paras=[]):
        [B, M, N, R, C, K, S, T, P] = Layer.getPara()
        [Tm, Tn, Tr, Tc, Tk] = (self.Tm, self.Tn, self.Tr, self.Tc, self.Tk)
        [W_p, I_p, O_p] = (self.W_p, self.I_p, self.O_p)

        if len(quan_paras)!=0:
            bits = quan_paras[0]+quan_paras[1]
            lat_W_mem = Tm * 1 * K * K / floor(W_p*16/float(bits))
        else:
            lat_W_mem = Tm * 1 * K * K / W_p

        lat_I_mem = min(Tn,N) * min(Tr,R) * min(Tc,C) / I_p
        lat_O_mem = min(Tm,M) * ceil(min(Tr,R) / S) * ceil(min(Tc,C) / S) / O_p

        if pattern_ones!=-1:
            lat_Comp = pattern_ones * ceil(min(Tr,R+P*2) / S) * ceil(min(Tc,C+P*2) / S)
        else:
            lat_Comp = K * K * ceil(min(Tr,R+P*2) / S) * ceil(min(Tc,C+P*2) / S)

        Lat1 = max(lat_I_mem, lat_W_mem, lat_Comp)
        Lat2 = max(ceil(N / Tn) * Lat1, lat_O_mem)
        Lat = B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm) * Lat2  # + (tO_mem + Lat1)

        if Lat2 == lat_O_mem:
            bottle_neck = "storing OFM"
        elif Lat1 == lat_I_mem:
            bottle_neck = "loading IFM"
        elif Lat1 == lat_W_mem:
            bottle_neck = "loading Weight"
        elif Lat1 == lat_Comp:
            bottle_neck = "computing"
        else:
            bottle_neck = "wired"

        I = lat_I_mem * ceil(N / Tn) * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)
        O = lat_O_mem * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)
        W = lat_W_mem * ceil(N / Tn) * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)
        C = lat_Comp * ceil(N / Tn) * B * ceil(R / Tr) * ceil(C / Tc) * ceil(M / Tm)

        return Lat, bottle_neck, [I,O,W,C]

    def get_layer_latency(self, Layer, pattern_ones=-1, quan_paras=[]):
        if self.layer_template_match_check(Layer):
            if self.T == "cconv":
                return self.get_cconv_latency(Layer, pattern_ones, quan_paras)
            else:
                return self.get_dconv_latency(Layer, pattern_ones, quan_paras)
        else:
            sys.exit(0)




if __name__ == '__main__':

    [B,M,N,R,C,K,S] = (config.B, config.M, config.N, config.R,config.C,config.K,config.S)
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = (config.Tm, config.Tn, config.Tr, config.Tc, config.Tk, config.W_p, config.I_p, config.O_p)


    [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (1024, config.DSP_BOUND, config.BRAM_BOUND, config.BRAM_SIZE, config.BITWIDTH)

    Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv")

    acc_1 = FPGA_Templates(Tm, Tn, Tr, Tc,
                         Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH)

    print(acc_1.get_layer_latency(Layer))


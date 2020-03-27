import sys


class Layer_Class:
    def __init__(self, B, M, N, R, C, K, S, T, P=0):
        self.B = B # batch size
        self.M = M # output channel
        self.N = N # input channel
        self.R = R+P*2 # output row
        self.C = C+P*2 # output col
        self.K = K # weight kernel size
        self.S = S # stride

        if T=="cconv" or T=="dconv":
            self.T = T
        else:
            print("only support conventional conv2d (cconv) and depthwise conv (dconv)")
            sys.exit(0)

    def printf(self):
        print(self.B, self.R, self.C, self.N, self.K, self.M, self.S, self.T)

    def getPara(self):
        return float(self.B), float(self.M), float(self.N), float(self.R), float(self.C), \
               float(self.K), float(self.S), self.T

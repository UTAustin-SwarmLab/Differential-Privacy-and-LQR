import numpy as np
import _pickle as cPickle
import traceback

src_file_root = "./src/"

class src:
    def __init__(self, T, deltaS, alpha, beta, gamma, name) -> None:
        self.name = name

        # try:
        #     self.Load()
        # except:
        ### Parameter for logistic function and DP
        self.T = T
        self.deltaS = deltaS ### sensitivity
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.covar = None
        self.forecaster = None
        # self.Save()

        return

    def GenGaussian(self, scale, seed) -> None:
        np.random.seed(seed)
        A = np.random.rand(self.T, self.T)
        B = A / np.amax(A)
        self.covar = np.cov(B)
        # print(self.covar/np.amax(self.covar))

        return

    def MeasureErrorCovar(self, past, ground_truth):
        predict = self.forecaster.predict(past)
        error = ground_truth - predict
        self.covar = np.cov(error, rowvar=False)

        return

    def Save(self):
        file = open(src_file_root+self.name+'.txt','wb')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

        return

    def Load(self):
        file = open(src_file_root+self.name+'.txt','rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)
        
        return

import numpy as np
import os
#%matplotlib
import matplotlib.pyplot as plt
class Mcmcplot:

    wdata =np.ones((1,1))

    def initialiseweights(self,numsamples,weightlength):
        self.wdata = np.ones((numsamples,weightlength))


    def addweightdata(self,sampleindex,w):
        self.wdata[sampleindex]=np.array(w)


    def saveplots(self):
        #for column in wdata:
        n_bins = 30
        tempw = self.wdata.transpose()
        for i,item in enumerate(tempw):
            plt.hist(item,bins=n_bins)
            if os.path.exists('mcmcplots') is False:
                os.makedirs('mcmcplots')
            name = 'mcmcplots/W'+str(i)+'.png'
            plt.savefig(name)
            plt.clf()

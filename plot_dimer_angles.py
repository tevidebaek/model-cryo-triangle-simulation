import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    def getAngleDist(filename):
        data = np.load(filename)

        #order of data is [dx, dy, dz, th_roll, th_twist, th_bend]
        
        
        #first we want to make a cut based on the interaction distance in the x-direction
        xbins = np.linspace(-1,1,11)
        
        plt.figure()
        plt.hist(data.T[0], bins=xbins)
        plt.show()
        

        cut_interaction = np.where((data.T[0]>0.25)&(data.T[0]<0.75))
        #cut_interaction = np.where((data.T[0]>-0.75)&(data.T[0]<-0.25))
        

        th_bend = data.T[-1][cut_interaction]*180/np.pi
        th_twist= data.T[-2][cut_interaction]*180/np.pi
        th_roll = data.T[-3][cut_interaction]*180/np.pi

        return th_bend, th_twist, th_roll

    filename1 = './SimulationOutput/Side2/run_003/side2_fluctuations.npy'
    #filename2 = './SimulationOutput/Side2/run_002/side2_fluctuations.npy'
    filename2 = './SimulationOutput/Side2/run_002/side2_fluctuations.npy'
    filename3 = './SimulationOutput/Side3/side3_fluctuations.npy'

    B1, T1, R1 = getAngleDist(filename1)
    B2, T2, R2 = getAngleDist(filename2)
    B3, T3, R3 = getAngleDist(filename3)

    #plot the bending angles
    fig, axs = plt.subplots(3,1, figsize=(6,5))

    bins = np.linspace(-50, 50, 41)

    axs[0].hist(B1, bins=bins, ec='k', fc='blue')
    axs[1].hist(B2, bins=bins, ec='k', fc='yellow')
    axs[2].hist(B3, bins=bins, ec='k', fc='red')

    plt.show()
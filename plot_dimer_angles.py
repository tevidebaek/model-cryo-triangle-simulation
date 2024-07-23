import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    filename = 'side1_fluctuations.npy'

    data = np.load(filename)

    #order of data is [dx, dy, dz, th_roll, th_twist, th_bend]

    #for now plot the bend angle

    for i in range(6):

        plt.figure()
        plt.hist(data.T[i], bins=20)
    plt.show()
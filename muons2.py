import os 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
import math as m 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
from muons1 import B1
from scipy.optimize import curve_fit

def std_dependence():        
    N = np.arange(5,10001,1)
    err = np.zeros(N.shape)
    for i in range(len(N)):    
        x = B1()
        x.read(N[i])
        err[i] = x.derive_std(x.NLL,[0.3,0.4,0.5],0.3)        
    plt.figure()
    plt.title("std N dependence")
    plt.xlabel("Number of pairs")
    plt.ylabel("error, (picoseconds)")
    plt.ylim(ymax=0.2)
    plt.ylim(ymin=-0.01)    
    plt.axhline(y=0.001)
    plt.plot(N,err,'*')
       
    def ffunc2(x, a):
        return a/np.sqrt(x)        
    popt, pcov = curve_fit(ffunc2, N, err, p0=(1)) 
    nN =np.arange(5,300000)
    plt.plot(nN, ffunc2(nN,popt[0]))
    plt.grid()
    plt.show()
   
 
#####      UNCOMMENT BELOW FOR QUICK-START
#std_dependence()  


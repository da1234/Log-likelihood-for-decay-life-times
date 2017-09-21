# -*- coding: utf-8 -*-
import os 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
import math as m 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
from scipy.integrate import trapz 






class B1:
  
    
    def __init__(self):

        self.life_time_pairs = [] 
    
 
        
    def read(self,lim=1e4):
        
        cnt=0
        
        with open('/Users/darrelladjei/python/ComPhys/projectB1/lifetime.txt','r') as f: 
            for line in f:
                if cnt<lim:
                    pairs = line.split()
                    self.life_time_pairs.append(pairs)
                    cnt+=1
                
                
                
        self.T = np.zeros(len(self.life_time_pairs))
        self.sigma = np.zeros(len(self.life_time_pairs))
                
                
        for i,j in enumerate(self.life_time_pairs):
            
            self.T[i] = float(j[0])
            self.sigma[i] = float(j[1])
    
        
    
    
    
    
    def hist_plot(self):
        
        time_freq, time_bin_edges = np.histogram(self.T, bins = 1000)
        sigma_freq, sigma_bin_edges = np.histogram(self.sigma,bins = 1000)
        
        pdf_values = []
        
                
        tau = 0.4
        
        
        for t,s in zip( sorted(self.T),sorted(self.sigma)):
    
            pdf_values.append(self.pdf_value(tau,s,t,BACKGROUND=False,a=0))
            
                   
        pdf_values= np.asarray(pdf_values)

        plt.figure()
        plt.title("Time Histogram")
        plt.xlabel("Time, t (picoseconds)")
        plt.ylabel("Probability density, PD")
        plt.hist(self.T, time_bin_edges,normed= True,histtype='step',color='k')
        
        plt.plot(sorted(self.T), pdf_values, label ='PDF',color = 'b')
        plt.show()
        
        plt.figure()
        plt.title("Error Histogram")
        plt.xlabel("Error, (picoseconds)")
        plt.ylabel("Frequency, f")
        plt.hist(self.sigma, sigma_bin_edges)
        plt.show()
        
        
    def background(self,time,sigma):
        
        Coeff = (1./(sigma*(m.sqrt(2.*m.pi))))
        exp = np.exp(-0.5*(time**2/sigma**2))
        
        return Coeff*exp
        
        
    def integrate(self):
        
        tau = np.mean(self.T)
        sig = np.mean(self.sigma)
        
        one = self.pdf_value(tau,sig,sorted(self.T))
        
        
        area = trapz(one,sorted(self.T))
        
        print area
        
        
        
    
    def pdf_value(self,tau,sigma,time,BACKGROUND=False,a=0):
        
        if not BACKGROUND:
        
            part1 = 0.5*(1./tau)*np.exp((sigma**2)/((tau**2)*2.) - (time/tau))
            part2= sp.erfc( (1./m.sqrt(2))*((sigma/tau) - (time/sigma)) )
                
            pdf_value = part1*part2
            
        else:
            
            pdf_value = a*self.pdf_value(tau,sigma,time) + (1-a)*self.background(time,sigma)
            
        
        return pdf_value
        
    



    def pdf_plot(self,BACKGROUND=False,a=0):
        
        tau = 0.4
        
        pdf_values = []
        
        
        for t,s in zip( sorted(self.T),sorted(self.sigma)):
    
            pdf_values.append(self.pdf_value(tau,s,t,BACKGROUND,a))
            
                   
        pdf_values= np.asarray(pdf_values)
        
        plt.figure()
        plt.title("pdf graph")
        plt.xlabel("time, picoseconds")
        plt.ylabel("probability density function")
        plt.plot(sorted(self.T), pdf_values, label ='PDF')
        plt.show()
        
        
    
    def NLL(self,tau,a=0,BACKGROUND=False):

        
        NLL_value=0

        NLL_value = -np.sum(np.log(self.pdf_value(tau,self.sigma,self.T,BACKGROUND,a)))
        
        return NLL_value
        
    def NLL_3dPlot(self,tau,bACKGROUND,a,function,probe=False):
        
        TAU,A = np.meshgrid(tau,a)
        ls= np.array([function(tau=t,BACKGROUND=True,a=frac) for t,frac in zip(np.ravel(TAU),np.ravel(A)) ])
        LS =ls.reshape(TAU.shape)
        
        fig =plt.figure()
        ax = fig.add_subplot(111,projection ='3d')
        ax.plot_surface(TAU,A,LS, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.0, antialiased=True)
          
        ax.set_xlabel('tau, (picoseconds)')
        ax.set_ylabel('a')
        ax.set_zlabel('nll')
        
        
        
        minimum_pairs = self.gradient_minim(function,tau_ini=0.3,a_ini=0.9,tolerance=1e-5)
        NLL_min = minimum_pairs[1]
        tau_min = minimum_pairs[0][0]
        a_min = minimum_pairs[0][1]
        
        
        ##for contour 
        plt.figure()
        
        if probe:
            cp = plt.contour(TAU,A,LS,levels=[NLL_min+0.5])
            p = cp.collections[0].get_paths()[0]
            
            print NLL_min
            v = p.vertices
            tau_range = v[:,0]
            a_range = v[:,1]
            
            print "tau range is", tau_range
            
            ##for the std or each variable
            tau_plus = max(tau_range)
            tau_minus = min(tau_range)
            std_tau_plus = tau_plus-tau_min
            std_tau_minus = tau_min-tau_minus
            
            
            
            ##for the std or each variable
            a_plus = max(a_range)
            a_minus = min(a_range)
            std_a_plus = a_plus-a_min
            std_a_minus = a_min-a_minus        
                
            print "boundaries are", std_tau_plus,std_tau_minus,std_a_plus,std_a_minus
            
        else:
            cp = plt.contour(TAU,A,LS)
            p = cp.collections[0].get_paths()[0]
            
            v = p.vertices
            tau_range = v[:,0]
            a_range = v[:,1]
            
            
            
        
        plt.clabel(cp,inline=True,fontsize=10)
        plt.title('Contour plot of tau and a')
        plt.xlabel("tau, (picoseconds)")
        plt.ylabel("a")        
  
        
        plt.show()
        
    
        
        
    def NLL_plot(self,taus=np.array([None]),BACKGROUND=False,a=np.array([None])):
        
        NLL_values=[]
        
        
        if not taus.all():
            taus = np.zeros(10)
            taus.fill(0.001)
            
        if BACKGROUND==False:
            a = np.ones(taus.size)           
            
        
        for pos,tau in enumerate(taus):
      
            NLL_values.append(self.NLL(tau,BACKGROUND))
            print self.NLL(tau,BACKGROUND)
            
        plt.figure()
        plt.grid()
        plt.title("NLL graph")
        plt.xlabel("average decay time estimation,(picoseconds)")
        plt.ylabel("negative log-likelihood")
        plt.plot(taus, NLL_values)
        #plt.plot(a, NLL_values)
        plt.show()
        
     


    def tol(self,a,b,tolerance):
        
        ans = False
        
        acc = (b-a)/b
        
        if acc <=tolerance:
            
            ans = True 
            
        return ans 
        
        
            
    
    def parab_minimiser(self,function,start_pt, tolerance):
        
  
        
        ##to store values 
        x_vals = sorted(start_pt)
        a = x_vals[0]
        b = x_vals[2]
        
        
        while not self.tol(a,b,tolerance):
            
        
            x0 = x_vals[0]
            x1 =x_vals[1]
            x2 = x_vals[2]  
            

            x3_num = (x2**2 - x1**2)*(function(x0)) + (x0**2 -x2**2)*(function(x1)) + (x1**2 -x0**2)*(function(x2))
            x3_den = (x2 - x1)*(function(x0)) + (x0 -x2)*(function(x1)) + (x1 -x0)*(function(x2))
            x3 = 0.5 *(x3_num/x3_den)
            
        
            
            all_x_vals = [x0,x1,x2,x3]
            y_vals=np.zeros(len(all_x_vals))
        
            
            
            for pos, val in enumerate(all_x_vals):
                
                y_vals[pos]= function(val)
            
                
            arg_max = np.argmax(y_vals)
            x_vals= all_x_vals[:arg_max] + all_x_vals[arg_max+1:] 
            
            
            arg_sort = np.argsort(x_vals)           
            a = x_vals[arg_sort[0]]
            b = x_vals[arg_sort[1]]
            
        return a, x_vals
      
            
                  
    def find_std(self,function,start_pt, tolerance):
        
        minimum = self.parab_minimiser(function,start_pt, tolerance)[0]
        
        print "min is", minimum
        
        min_val = function(minimum)
        
        val_scope_min = minimum-(minimum*0.1)
        val_scope_max = minimum+(minimum*0.1)
        
        val_scope = np.linspace(val_scope_min,val_scope_max,num=100)
         
        
        self.NLL_plot(val_scope)
        
        values = []
        
        for pt in val_scope:
            
            val = function(pt)
            
                        
            if abs(val-min_val) < 0.5:
                
                values.append(pt)
        
        tau_minus = values[0]
        tau_plus = values[-1]
        print "std above and below is",tau_minus-minimum, tau_plus-minimum
        
    def derive_std(self,function,start_pt, tolerance):
        

        
        x_vals = self.parab_minimiser(function,start_pt, tolerance)[1]
        
        x2 = x_vals[2]
        x1 = x_vals[1]
        x0 = x_vals[0]
        
        d = (x1-x0)*(x2-x0)*(x2-x1)
        
        P2 = (2/d)*((x2-x1)*function(x0) + (x0-x2)*function(x1) + (x1-x0)*function(x2))
        std = 1/m.sqrt(P2)
        
        print "derived std gives", std
        
        return std 
        
        
        
        
        
    
    def gradient_minim(self, function,tau_ini,a_ini,tolerance,h=1e-5,alpha=1e-5):
        
        
        x0 = np.array([tau_ini,a_ini])
        y0 = function(tau=x0[0],a=x0[1],BACKGROUND=True)
        
        tau_grad  = (1./h)*(function(x0[0]+h,x0[1],BACKGROUND=True) - function(x0[0],x0[1],BACKGROUND=True))
        a_grad  = (1./h)*(function(x0[0],x0[1]+h,BACKGROUND=True) - function(x0[0],x0[1],BACKGROUND=True))
        
        grad = np.array([tau_grad,a_grad])
        
        x1 = x0 - alpha*grad
        
        y1 = function(x1[0],x1[1],BACKGROUND=True)
        
        
        diff = abs(y1-y0)
        
        while diff > tolerance:
            
            x0 = x1
            y0 = function(tau=x0[0],a=x0[1],BACKGROUND=True)
            tau_grad  = (1./h)*(function(x0[0]+h,x0[1],BACKGROUND=True) - function(x0[0],x0[1],BACKGROUND=True))
            a_grad  = (1./h)*(function(x0[0],x0[1]+h,BACKGROUND=True) - function(x0[0],x0[1],BACKGROUND=True))
            grad = np.array([tau_grad,a_grad])
            x1 = x0 - alpha*grad
            y1 = function(x1[0],x1[1],BACKGROUND=True)
            diff = abs(y1-y0)
            
            
        print x1,y1
        return x1,y1
        
        
        
        
        
                

                

 #####      UNCOMMENT BELOW FOR QUICK-START      
                  
    
#x = B1()
#x.read()
#x.integrate()
#x.hist_plot()
#x.NLL_plot(taus=np.linspace(0.3,0.5,100))
#x.parab_minimiser(x.NLL,[0.3,0.34,0.5],1e-5)
#x.derive_std(x.NLL,[0.3,0.4,0.5],1e-5)
#x.NLL_3dPlot(np.linspace(0.3,0.5,100),True,np.linspace(0.9,1.0,100),x.NLL,probe=True)
#x.find_std(x.NLL,[0.3,0.4,0.5],1e-5)
#x.gradient_minim(x.NLL,tau_ini=0.4,a_ini=0.9,tolerance=1e-5)

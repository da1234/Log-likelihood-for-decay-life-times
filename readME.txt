Hi, Follow the instructions bellow 

FOR QUICK-START:

		Ready made functions as the bottom of each file - uncomment and run 

import muons1.py 

Make an instance of B1
	
takes no arguments 

execute the read() function:

			takes one argument:
					
					length: the number of readings to use, 
						defaults to 10,000


functions available:


			hist_plot():
					takes no arguments
					plots data as a histogram with fit function over it (no background)

			integrate():
					takes no arguments 

					returns integral of PDF (no background)



			pdf_value():
					arguments:

						tau - average lifetime
						
						sigma - error in tau

						time 

						BACKGROUND - boolean, True for background inclusion 

						a - fraction of signal in data set

					returns PDF value


			pdf_plot():
					arguments:


						BACKGROUND 

						a 

					returns plot of PDF

			pdf_value():
					arguments:

						tau - average lifetime
						
						sigma - error in tau

						time 

						BACKGROUND - boolean, True for background inclusion 

						a - fraction of signal in data set

					returns PDF value





			NLL():
					arguments:

						tau						
						
						
						BACKGROUND 

						a 

					returns NLL value 






			NLL_3dPlot():

						arguments:


						tau 
		

						BACKGROUND 

						a 

						function - function to minimise - should be NLL

						probe- boolean, only plots contour at NLL + 0.5

					returns contour Plot and 3d plot 

			NLL_Plot():

						arguments:


						taus - array of tau values
		

						BACKGROUND 

						a 

	

					returns NLL plot 


			parab_minimiser():

						arguments:


						function - function to minimise - should be NLL

						start_pt = array - starting point

						tolerance - tolerance

					returns minimum point, last three tau values


			find_std():

						arguments:
	

						start_pt 


						function - to minimise - should be NLL

						tolerance

					returns standard deviation form NLL+0.5


			derive_std():

						arguments:


						function 
		

						start_pt 

						tolerance


					returns error from curvature 

			gradient_minim():

						arguments:


						function  
		

						tau_ini 

						a_ini

						tolerance
 

					returns minimum point, NLL value at minimum  
 
 
 
For error v. N dependence

import muons2.py

			run std_dependence()

			returns graph of error against N







					


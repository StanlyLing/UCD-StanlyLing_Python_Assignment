### Python First Assignment
### Stanly Yan Seng Ling
### Space Detector Lab 1
### Gamma Ray Burst Spectrum Analysis


################################# Import Important Library Packages ##########################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import inf as INF
import argparse

################################# Define Usable Constant #####################################

TWO_PI = np.pi * 2.
Gaussian_bounds = [

    np.array([0, 0, 0]),
    np.array([INF, INF, INF])
]

Gaussian_bounds_background = [

    np.array([0, 0, 0, -INF, 0]),
    np.array([INF, INF, INF, INF, INF])
]

Gaussian_params = ('mu', 'sig', 'a', 'c0', 'c1')

# Region of Interest from the full spectrum (ROI)

ROI = (6640, 6750)

##################################### Read Full Raw Spectrum data#############################

def read_raw_spectrum(filename):

    spectrum_data = []
    
    with open(filename,'r') as file:
        
        for line in file:

            line = line.strip()
            
            if line.startswith('#'):
                next(file)
                continue

            elif not line[0].isdigit():
                continue
                
            else:
                wavelength, flux = line.split(',')
                spectrum_data.append((float(wavelength), float(flux)))

    return spectrum_data


########################## Extracting Data from Spectrum_data list ###########################

def wavelengths_and_fluxes(filename):

    data = read_raw_spectrum(filename)

    #COnverting data into lists

    wavelengths = [point[0] for point in data]
    fluxes      = [point[1] for point in data]
    
    return wavelengths, fluxes


################################# Plotting the Full Spectrum #################################

def plot_the_spectrum(wavelengths, fluxes):
    
    #Plotting the full spectrum
    plt.plot(wavelengths, fluxes, label = "Spectrum Data", color = 'g')
    plt.legend()
    plt.title('Wavelengths VS Fluxes Spectrum (Full Spectrum)')
    plt.xlabel('Wavelengths')
    plt.ylabel('Fluxes')
    plt.grid(True)


################################## Define Gaussian Profile ###################################

def gaussian_first_fit(x, mu, sig, a):

    return a * np.exp(-0.5 * (x-mu)**2 / sig**2) / np.sqrt(TWO_PI * sig**2)


################################# Define the interested region ###############################

def in_interval(x, xmin = -INF, xmax = INF):

    """
    Boolean Mask with Value True for x in xmax and xmin
    """

    _x = np.asarray(x)

    return np.logical_and(xmin <= _x, _x < xmax)

def filter_in_interval(x, y, xmin, xmax):

    """
    Select only elements of x and y where xmin <= _x_wavelength < xmax
    """
    # Check if X and y lists has the same shape
    # Note .shape is only attributed and used with numpy array, not lists
    # Check the type of data before checking their shape 

    x = np.asarray(x)
    y = np.asarray(y)
    
    print("\n\nChecking the dimensions of both X and Y data ......")

    if x.shape != y.shape:
    
        print("X and Y lists has no same shape, check your code\n")

    else:
		
        print("X-data and Y-data have the same shape, you are good to proceed!\n")

    _mask = in_interval(x, xmin, xmax)
            
    return [np.asarray(a)[_mask] for a in (x,y)]

########################## Define Interested Region With Desire Colour #######################

def colourmask(x, xmin = -INF, xmax = INF, cin = 'red', cout = 'gray'):

    """
    This is to colour cin if within the region of interest
    """
    _mask = np.array(in_interval(x, xmin, xmax), dtype = int)

    #Convert to colours 
    colourmap = np.array([cout, cin])
    return colourmap[_mask]

################################## Plot Interested Region ##################################

def plot_interested_region(filename):

    #Converting data from lists to numpy array
    wavelengths, fluxes = wavelengths_and_fluxes(filename)
    
    plot_the_spectrum(wavelengths, fluxes)

    colours = colourmask(wavelengths, xmin = 6680, xmax = 6700)
    plt.scatter(wavelengths, fluxes, c = colours, marker = '+', label = 'Region of Interest(ROI)' )

    #plt.legend()
    #plt.show()

###################### Define Curve fit and their key input arguments #######################

def simple_model_fit(model, wavelengths, fluxes, roi, **kwargs):

    """
    Least Square Estimate of model parameters.
    """

    #Select relevant channels and counts
    _wavelengths, _fluxes = filter_in_interval(wavelengths, fluxes, *roi)

    #Fit the model to the data
    popt, pcov = curve_fit(model, _wavelengths, _fluxes, **kwargs)

    return popt, pcov

##################### Define the Output Format of the Final Optimal Value ####################

def format_result(params, popt, pcov):

    """
    Display parameter best estimates and uncertainties
    """

    #Extract the uncertainties from the covariance matrix
    perr = np.sqrt(np.diag(pcov))

    #Format parameters with best estimates and uncertainties

    _lines = (f"{p} = {o:.4f} ± {e:.4f}" for p,o,e in zip(params, popt, perr))

    return "\n".join(_lines)

################# Define the plotting model of Gaussian and Continuum Profile ################

def plot_model(ax, model, xrange, ps, npoints=1001, **kwargs):

    """
    Plots an 1d model on an Axes smoothly over xrange
    """
    _wavelengths = np.linspace(*xrange, npoints)
    _fluxes      = model(_wavelengths, *ps)
    
    return ax.plot(_wavelengths, _fluxes, **kwargs)

################### Define Initial Estimate value for Gaussian Profile ######################

def first_moment(x, y):

    x = np.asarray(x)
    y = np.asarray(y)
    
    return np.sum(x*y) / np.sum(y)

def second_moment(x, y):

    x = np.asarray(x)
    y = np.asarray(y)
    x0 = first_moment(x, y)

    return np.sum(((x-x0)**2) *y) / np.sum(y)

def gaussian_initial_estimates(filename, wavelengths, fluxes):

    """
    Estimates of three parameters of the gaussian distribution
    """

    #Initial Guess centroid of the peak wavelength
    mu0  = first_moment(wavelengths, fluxes)

    #Initial Estimates Standard Deviation
    sig0 = np.sqrt(second_moment(wavelengths, fluxes))

    #Initial Estimates Amplitude
    a0 = np.max(fluxes)
    
    c0, c1 = continuum_point(filename)

    return (mu0, sig0, a0, c0, c1)

################################# Define the Continuum Line #################################

# Subtracting Background Components

def background_slope(x, c0, c1):

    return c1*x + c0

################################# Plotting Continuum Line ##################################

def continuum_plot(filename):

    fig, ax = plt.subplots(figsize=(9,5))
    plot_interested_region(filename)
    wavelengths, fluxes = wavelengths_and_fluxes(filename)
    popt, pcov = curve_fit(background_slope, wavelengths, fluxes)
    
    plot_model(ax, background_slope, (6640, 6750), popt, linestyle = 'dashed', c = 'b', label = 'Continuum Fit')

    plt.legend()
    plt.title('Wavelengths VS Fluxes Spectrum (Continuum Fit Only)')

################################# Getting c0, c1 without plotting ############################

def continuum_point(filename):

    wavelengths, fluxes = wavelengths_and_fluxes(filename)
    popt, pcov = curve_fit(background_slope, wavelengths, fluxes)
    
    c0, c1 = popt[0], popt[1]
    
    return c0, c1

############################### Define Gaussian + Continuum Profile ##########################

def gaussian_plus_background(x, mu, sig, a, c0, c1):

    """
    This is the definition to illustrate a gaussian on a linear background line.
    """
    return gaussian_first_fit(x, mu, sig, a) + background_slope(x, c0, c1)

###################### Plotting the final Curve Fit (Gaussian + Continuum) ###################

def plot_curve_fit(filename, popt, pcov):

    fig, ax = plt.subplots(figsize = (9,5))
    plot_interested_region(filename) 	

    #Plot the curve fit with background
    popt_background = popt[3:]
    
    plot_model(ax, gaussian_plus_background, ROI, popt, linestyle = 'dashed', c='r', label='Gaussian-Continuum Fit')
    plot_model(ax, background_slope, ROI, popt_background, linestyle = 'dashed', c = 'b', label = 'Continuum Fit')
    
    plt.legend()
    plt.title('Wavelengths VS Fluxes (Gaussian-Continuum Fit)')
    plt.show()

####################### Gaussian fit with the substraction of background #####################

def optimal_value(filename, wavelengths, fluxes):

    c0, c1 = continuum_point(filename)
    continuum_plot(filename)

    #Make Initial Estimates or Guesses
    _wavelengths, _fluxes = filter_in_interval(wavelengths, fluxes, *ROI)
    _p0 = gaussian_initial_estimates(filename, wavelengths, fluxes)

    #Show the initial guess
    print("\n> The initial estimates:\n")
    print("\n".join(f"{p} = {o:.4f}" for p, o in zip(Gaussian_params, _p0)))

    #Do the fit 
    popt, pcov = simple_model_fit(gaussian_plus_background, wavelengths, fluxes, ROI, p0 =        [_p0[0], _p0[1], _p0[2], c0, c1] , bounds = Gaussian_bounds_background)

    return popt, pcov

######################### Define the main function to run everything #########################

def main(filename = 'spectrum.txt'): 

    parser = argparse.ArgumentParser(description = 'Fit Gaussian and Continuum Line to Full Spectrum' )
    parser.add_argument('filename', help = 'Input Raw Spectrum Data')
    args = parser.parse_args()
     
    # Extracting Spectrum data from txt file
    wavelengths, fluxes = wavelengths_and_fluxes(filename)

    #Display the result  
    plot_the_spectrum(wavelengths, fluxes) 
    popt, pcov = optimal_value(filename, wavelengths, fluxes)
    print("\n> The final fitted estimates:\n")
    print(format_result(Gaussian_params, popt, pcov))
    plot_curve_fit(filename, popt, pcov)


if __name__ == '__main__':

    main('spectrum.txt')









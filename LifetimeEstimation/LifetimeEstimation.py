###### Lifetime Estimation.py ######
#
# This class is intended to be used with reduced datasets from purification tests,
# serving as a means to estimate the electron lifetime of the liquid in the time
# projection chamber (TPC).
# 
# The class assumes a Poisson-distributed probability of an individual 
# recombination of a free electron in the medium of the TPC, thereby making the 
# recombination of all free electrons in the medium follow an exponential 
# distribution.
#
# The class works around the noise of the reduced datasets by fitting to a plot
# of total tile energy vs. drift time of the chamber, fitting datapoints to 
# slices of drift time in the whole dataset. In other words, the tile energy vs. 
# drift time is split into intervals of drift time (not equally-spaced intervals),
# and in each slice, the events are histogrammed according to tile energy, then
# fit to a Gaussian, the centroid of which is taken as the mean tile energy that 
# arrived in that drift slice. Those centroids are then used as the fit points 
# for the exponential distribution. The natural log of those points is taken for 
# a linear least-squares fit, the slope of which is the opposite of the reciprocal
# of the lifetime.
# 
# This class is intended to be used after quality cuts on Tile and SiPM energy are
# determined by the user to see clearly the alpha band present in purification data
# when plotted as tile energy vs. drift time. These quality cuts are inputs to the 
# fitting function.
# 
# Methods:
# 
# gauss(x, A, x0, sigma) - A standard Gaussian function that calculates 
#                          y = A*e^(-(x - x0)^2/(2*sigma^2)) for a dataset x and for
#                          parameters A, x0, and sigma. Returns the y array.
#
# gauss_fit(x, y) - A Gaussian fitting function that utilizes scipy.optimize.curve_fit()
#                   to fit x and y data to a Gaussian. Returns the parameters of the 
#                   Gaussian along with a covariance matrix.
#
# freedmanDiaconis(data, returnas="width") - A function that determines the "ideal" 
#                                            number of bins with which to histogram
#                                            a given dataset. Returns either the bin width
#                                            or the number of bins to use.
#
# lsLin(xs, ys) - A function that performs a linear, least-squares fit on a set of data (xs,
#                 ys). Returns the slope, intercept, their respective uncertainties, R^2,
#                 and the number of fit points.
#
# cutDriftSlices(various) - A function that takes the tile energy vs. drift time data from a 
#                           run, and with square quality cuts on tile and SiPM energy, bins
#                           the data into drift slices of non-uniform length based on what is
#                           statistically sound for histogramming/fitting to a Gaussian, and 
#                           returns those slices along with other information. Called within
#                           binnedElectronLifetimeFit().
#
# fitCentroidstoChargeData(various) - A function that takes a histogram with x and y data, selects
#                                     a subregion near the guessed centroid location corresponding
#                                     to the alpha band of the purification data, and fits a 
#                                     Gaussian to that data, returning the mean and std. dev of the 
#                                     fit along with other relevant statistics. Called within 
#                                     binAndFitChargeData(), which is itself called within
#                                     binnedElectronLifetimeFit().
#
# binAndFitChargeData(various) - A function that takes the tile energy vs. drift time data from a 
#                                run, and with linear cuts on the tile energy, square cuts on the
#                                SiPM energy, and slices of drift time, histograms the tile data
#                                in each slice, then fits those histograms to a Gaussian to extract
#                                the centroids for the lifetime fit, returning the centroids, their
#                                uncertainties, and other relevant statistics. Called within
#                                binnedElectronLifetimeFit().
#
# binnedElectronLifetimeFit(various) - The function that takes the data from a run, splits
#                                      it into drift slices, histograms those slices, fits
#                                      them to a Gaussian, then takes those centroids and 
#                                      fits the natural log of them to a line. Returns the 
#                                      left edges of the drift slices, the centroids of the
#                                      Gaussian fit, their uncertainties, the lifetime,
#                                      the intercept of the exponential fit, their uncertainties,
#                                      and the R^2 of the linear fit used with the log(centroids).
#
# NOTE/WARNING 1: There are some lines in binnedElectronLifetimeFit() that have values 
# that contain hard-coded numbers; these numbers were used empirically based on the
# reduced_v9 datasets from Run 34 and will likely need to be replaced by more robust
# and dynamic calculation methods/thresholds for other datasets and for more general
# usage. (The values that contain hard-coded numbers are threshold values for 
# conditional statements and the slice thickness gradient.)
#
# NOTE 2: The class utilizes somewhat centrally scipy.optimize.curve_fit() for the 
# Gaussian fitting and a hand-written function for the least-squares fit. In future
# revisions of this class, lmfit should be used in place of both of these functions, 
# just to get a better fit, better handling of all uncertainties, and a better
# visualization of the fit with all uncertainties involved. 
#
# ChangeLog:
# 2022-08-16 MM Created.
# 
###############################################################################################
#
# Load Classes/Modules necessary for the fits/plots

import pandas as pd                       # For handling the binary file
import numpy as np                        # For handling arrays, math
import matplotlib.pyplot as plt           # For plotting data
import matplotlib.patches as mpatches     # For adding ROIs to plots
import uproot                             # For ease of use with Pandas
import tables                             # For ease of use with Pandas
import array                              # For ease of use with Pandas
import cycler                             # For ease of use with Pandas
import histlite as hl                     # Not yet used, but for possible use with Histograms
import os                                 # For OS Commands
import sys                                # For changing the path environment variable
import time                               # For time access/conversions
import pickle                             # To read the binary files
from scipy.stats import norm              # For the Gaussian, not sure if used or not
from lmfit import minimize, Parameters    # To be used, not yet implemented (see NOTE 2)
from scipy.optimize import curve_fit      # Performs the Gaussian Fit
from scipy import stats as stat           # Used for Statistics on datasets

# Code written by Stanford for reading the binary files and for getting run info
from StanfordTPCAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration
from StanfordTPCAnalysis.WaveformAnalysis import Waveform

# Augment the path env variable
sys.path.append('/g/g20/lenardo1/software')

###############################################################################################

# gauss()
#
# gauss() takes an independent variable (array) and parameters of a Gaussian function and 
# returns the corresponding dependent variable array. This function is used in curve 
# fitting and in plotting. This function also has no y-offset parameter.
#
#                2         2
#       -(x - x0) /(2*sigma )
# y = Ae
#
# Inputs: 
# 
# x - a numpy array of values (floats), the "independent variable" of the function
#
# A - the scale factor of the Gaussian, a float
#
# x0 - the mean of the Gaussian, a float
#
# sigma - the standard deviation of the Gaussian, a float
#
# Outputs:
# 
# A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) - a numpy array of values (floats), the
#                                                 corresponding Gaussian of values given 
#                                                 the input x
#
# ChangeLog: 
# 2022-08-16 MM Created.

def gauss(x, A, x0, sigma):
    # Return output array as calculating the Gaussian array from inputs
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# gauss_fit()
#
# gauss_fit() takes an independent variable (array), known henceforth as an x array, and
# a dependent variable array, known henceforth as a y array, and fits a Gaussian curve to 
# the data, returning the parameters A, x0, and sigma from the fit for use in another 
# function, along with the covariance matrix of all parameters. 
#
# The function calls scipy.optimize.curve_fit() to fit the curve; in future revisions of 
# this function, lmfit should maybe be used for better quality fits/better handling of 
# uncertainties.
#
#                2         2
#       -(x - x0) /(2*sigma )
# y = Ae
#
# Inputs: 
# 
# x - a numpy array of values (floats), the "independent variable" of the data
#
# y - a numpy array of values (floats), the "dependent variable" of the data
#
# Outputs:
# 
# popt - a numpy array of parameters (floats) of the Gaussian fit on the data
#
# pcov - the covariance matrix (3 x 3, since there are 3 fit parameters) (float)
#        of the fit parameters
#
# ChangeLog: 
# 2022-08-16 MM Created.

def gauss_fit(x, y):
    # Calculate the mean guess for the curve_fit() function from data
    mean = sum(x * y) / sum(y)
    # Calculate the std. dev guess for the curve_fit() function from data
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    # Perform the fit, return the outputs
    popt, pcov = curve_fit(gauss, x, y, p0=[max(y), mean, sigma])
    return popt, pcov

# freedmanDiaconis()
#
# freedmanDiaconis() calculates an "ideal" or acceptable width of histogram bins
# for a given dataset based on the Freedman-Diaconis rule. 
#
# Paraphrased from Wikipedia: The Freedman-Diaconis rule is a rule that can be used
# to determine the number of bins for a histogram, based on minimizing (roughly) the
# integral of the squared difference between the relative frequency density and the 
# theoretical probability distribution (i.e. between the bin heights and the Gaussian).
# The bin width predicted by this rule is:
# 
#                bw = 2*IQR(x)/n^(1/3),
# 
# where bw is the bin width, n is the number of datapoints in a dataset/array x, and 
# IQR is the interquartile range of the data, namely the "middle 50%" of the data, or 
# the difference/width between the 25th and 75th percentiles of normally-distributed 
# data. In other words, the IQR of normally-distributed data spans from the mean 
# -0.6745*(std. dev) to the mean + 0.6745*(std. dev) of the data. See Wikipedia for 
# more information.
#
# This rule can be used as a suggestion for the number of bins with which to histogram;
# it is not set in stone, and in the lifetime estimation, the rule is not set in stone.
#
# Inputs:
#
# data - a numpy array of the data (floats) to be histogrammed
#
# returnas - a string that specifies what the user wants, the bin width or the 
#            number of bins to use in the histogramming
#
# Outputs - 
#
# bw - the bin width (float); if returnas is set to "width"
# 
# int((datrng/bw) + 1)  - the number of bins (int); if returnas is not set to "width"
#
# ChangeLog:
# 2022-08-16 MM Created.

def freedmanDiaconis(data, returnas="width"):
    # Set the input data as an array
    data = np.asarray(data, dtype=np.float_)
    # Calculate the interquartile range of the data
    IQR  = stat.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    # Get the data size 
    N    = data.size
    # Calculate the bin width from the IQR and data size
    bw   = (2 * IQR) / np.power(N, 1/3)

    # Return Values
    if returnas=="width":                         # return the width if specified
        result = bw                               
    else:                                         # if not, return the number of bins
        datmin, datmax = data.min(), data.max()   # extract the min and max of the data
        datrng = datmax - datmin                  # calculate the range of the data from min to max
        result = int((datrng / bw) + 1)           # calculate the number of bins from the range, width
    return(result)

# lsLin()
#
# lsLin() performs a linear least-squares fit on a dataset given both its x and y values.
# This function was written following the Least-Squares linear fit explanation on WolframAlpha
# basically line-for-line. Python likely has existing functions that perform this exact 
# operation.
#
# NOTE: The only quality of fit metric returned as of creation (16 August 2022) 
# of the function is R^2, the coefficient of determination. Future revisions should 
# include a chi-squared and reduced chi-squared metric, since all the pieces to calculate
# them are there.
#
# Inputs:
# 
# xs - a numpy array (floats) of the independent varaible values of the data
#      to be fit
#
# ys - a numpy array (floats) of the dependent variable values of the data to
#      be fit
# 
# Outputs: 
# 
# slope - the slope (float) of the linear fit
#
# sigma_slope - the uncertainty (float) of the slope of the linear fit
#
# intercept - the intercept (float) of the linear fit
#
# sigma_intercept - the uncertainty (float) of the intercept of the linear fit
#
# r_squared - the coefficient of determination (float) of the linear fit
#
# numel - the number of points (int) used in the fit
#
# ChangeLog:
# 2022-08-16 MM Created. 

def lsLin(xs, ys):
    
    # X Data Statistics
    square_x = np.square(xs)                                    # Squares of the x values
    second_moment_x = np.sum(square_x)                          # Second moment of the x dataset
    mean_x = np.mean(xs)                                        # Mean x value (first moment)
    mean_x_square = mean_x**2                                   # Square of the first moment
    numel = np.size(xs)                                         # Size of the dataset (i.e. number of fit points)
    ss_x = second_moment_x - numel*mean_x_square                # Sum of the squares of residuals of x (unscaled variance)
    var_x = ss_x/numel                                          # Variance of x
    
    # Y Data Statistics
    square_y = np.square(ys)                                    # Squares of the y values
    second_moment_y = np.sum(square_y)                          # Second moment of the y dataset 
    mean_y = np.mean(ys)                                        # Mean y value (first moment)
    mean_y_square = mean_y**2                                   # Square of the first moment
    ss_y = second_moment_y - numel*mean_y_square                # Sum of the squares of residuals of y (unscaled variance)
    var_y = ss_y/numel                                          # Variance of y
    
    # Covariance Stuff
    x_times_y = np.multiply(xs,ys)                              # Element-wise multiplication of x*y
    sum_x_times_y = np.sum(x_times_y)                           # Sum of the x*y element-wise multiplication 
    ss_xy = sum_x_times_y - numel*mean_x*mean_y                 # Sum of the squares of the residuals of both datasets (unscaled covariance)
    cov_xy = ss_xy/numel                                        # Covariance of x, y
    
    # Least Squares Calculations
    slope = cov_xy/var_x                                        # Slope calculation
    intercept = mean_y - slope*mean_x                           # Intercept calculation
    
    # R-Squared, Uncertainties of Parameters
    r_squared = ss_xy**2/(ss_x*ss_y)                            # R^2 calculation
    s = np.sqrt((ss_y-(ss_xy**2/ss_x))/(numel-2))               # Some sort of scale factor for uncertainties calculations
    sigma_slope = s/np.sqrt(ss_x)                               # Uncertainty of Slope
    sigma_intercept = s*np.sqrt((1/numel)+(mean_x**2/ss_x))     # Uncertainty of Intercept
    
    # Return all relevant values
    return slope, sigma_slope, intercept, sigma_intercept, r_squared, numel

# cutDriftSlices()
#
# cutDriftSlices() is a called function within binnedElectronLifetimeFit() that calculates
# various values concerning the range, slices, and intervals of drift time in both µs and
# in samples to be used later in the function, both for calculations and for printed reports
# and for plots. There is nothing physically significant about this function; it is merely
# a modular function that breaks up the size of the main fitting function, hence its 
# multiple inputs and outputs.
#
# Inputs: 
#
# data_df - the pandas dataframe file (hdf5) that contains all the events and information
#           about the events from a run
#
# charge_energy - the numpy array (float) that is taken from data_df that contains all tile
#                 energy for each event
#
# time_of_max_channel - the nupmy array (float?) that is taken from data_df that contains all
#                       the timestamp information (in samples) for each event
#
# analysis_config - an object that contains run parameters for certain conversions/calculations
#
# start_drift - the user-specified left edge (float) of the lifetime fitting range in µs
#
# end_drift - the user-specified right edge (float) of the lifetime fitting range in µs
#
# sipm_lower_bound - the user-specified lower boundary (float) of SiPM energy of all events in 
#                    ADC counts
#
# sipm_upper_bound - the user-specified upper boundary (float) of SiPM energy of all events in 
#                    ADC counts
#
# charge_lower_bound - the user-specified lower boundary (float) of charge tile energy of all events 
#                      in ADC counts
#
# charge_upper_bound - the user-specified lower boundary (float) of charge tile energy of all events 
#                      in ADC counts
#
# verbose - a flag (boolean) that specifies if results are desired to be printed; set to "True" by default
#
# Output: 
#
# drift_time - the nupmy array (float) that is taken from data_df that contains all
#                       the timestamp information (in µs) for each event
# 
# drift_slices - the numpy array (float) that contains the left edges of all drift time slices 
#                (in µs) in which centroids will be fit for the lifetime fit
#
# time_max_chan_bins - the numpy array (float) that contains the left edges of all drift time slices 
#                      (in samples) in which centroids will be fit for the lifetime fit
#
# drift_intervals - the numpy array (float) that contains the amount of time (in µs) between the
#                   left edges of each drift slice
#
# time_max_chan_intervals - the numpy array (float) that contains the amount of time (in samples) between the
#                   left edges of each drift slice
#
# iqr_drift_time - the interquartile range (float) of the tile energy data that has been cut 
#                  to specify the ideal number of drift slices by the Freedman-Diaconis rule
#
# dt_start_samples - the user-specified left edge (float) of the lifetime fitting range, converted
#                    to samples
#
# last_time_max_chan_interval - the last element (float) in time_max_chan_intervals
#
# ChangeLog:
# 2022-08-16 MM Created.

def cutDriftSlices(data_df, charge_energy, time_of_max_channel, analysis_config, start_drift, end_drift,\
                   sipm_lower_bound, sipm_upper_bound, charge_lower_bound, charge_upper_bound, verbose=True):
    
    # Convert the samples array to drift times in µs
    drift_time = (time_of_max_channel - analysis_config.run_parameters['Pretrigger Length [samples]']) \
               * analysis_config.run_parameters['Sampling Period [ns]'] / 1000.
    
    # Convert the left edge of the lifetime fit range to samples
    dt_start_samples = (1000/analysis_config.run_parameters['Sampling Period [ns]'])*start_drift \
                    + analysis_config.run_parameters['Pretrigger Length [samples]']
    
    # Convert the right edge of the lifetime fit range to samples
    dt_end_samples = (1000/analysis_config.run_parameters['Sampling Period [ns]'])*end_drift \
                    + analysis_config.run_parameters['Pretrigger Length [samples]']
    
    # Create Drift Time + Corresponding Sample Bins from Start and End Ranges + Interval
    
    # Create a mask of the data (square in SiPM energy, tile energy, and drift time) that visualizes the alpha band (determined by user)
    dt_bins_mask = (data_df['TotalSiPMEnergy']> sipm_lower_bound) & (data_df['TotalSiPMEnergy']< sipm_upper_bound) \
            & (data_df['NumTileChannelsHit'] < 3) & (data_df['IsFull3D']) \
            & (data_df['TotalTileEnergy']> charge_lower_bound) & (data_df['TotalTileEnergy']< charge_upper_bound) \
            & (data_df['TimeOfMaxChannel']> dt_start_samples) & (data_df['TimeOfMaxChannel']< dt_end_samples)
    
    # Calculate the IQR of the charge energy data with this mask applied for future calculations in the fit function
    iqr_drift_time = stat.iqr(charge_energy[dt_bins_mask],rng=(25, 75),scale=1.0,nan_policy="omit")
    
    # Apply a Gradient for Drift Time Slice Thicknesses and Slices:
    
    # Determine the "equally-spaced" optimal slice thickness which will be scaled for later slices
    original_slice_width = (end_drift - start_drift)/freedmanDiaconis(charge_energy[dt_bins_mask],returnas="fuck")
    
    # Print some values thus far
    if verbose:
        print(f'Number of Events with Mask for DT Slices: {np.size(charge_energy[dt_bins_mask])}')
        print(f'IQR of Charge Energy with Preliminary Mask: {iqr_drift_time}')
        print(f'Evenly-Spaced Drift Slice Thickness based on Freedman_Diaconis Rule: {original_slice_width} µs')
        
    # Set the Slice Gradient: THIS IS HARD-CODED RIGHT NOW!!!!!
    slice_gradient = 1.04 # Scale Factor
    
    # Create the drift slices according to this scaling (inefficient):
    
    # First make a ton of them
    drift_slices = np.array([start_drift + (slice_gradient**x)*x*original_slice_width for x in range(end_drift-start_drift)])
    
    # Now keep only the ones that end before the user-specified right edge of the fit range
    drift_slices = drift_slices[np.where(drift_slices<end_drift)]
    
    # Convert these to samples for functional use
    time_max_chan_bins = (1000/analysis_config.run_parameters['Sampling Period [ns]'])*drift_slices \
                    + analysis_config.run_parameters['Pretrigger Length [samples]']
    
    # Create the last drift interval slice, which doesn't get created in the loop that created all the others
    last_drift_interval = np.array([(start_drift+(slice_gradient**len(drift_slices))*len(drift_slices)*original_slice_width)-drift_slices[-1]])
    
    # Add this last slice to the others - these are the intervals of time between drift slice left edges (non-uniform)
    # These also get used for plots
    drift_intervals = np.concatenate((np.diff(drift_slices),last_drift_interval))
    
    # Convert these intervals to samples for functional use
    time_max_chan_intervals = np.array([time_max_chan_bins[x+1]-time_max_chan_bins[x] for x in range(len(time_max_chan_bins)-1)])
    
    # Create the last sample interval slice as was done for the drift times
    last_time_max_chan_interval = (1000/analysis_config.run_parameters['Sampling Period [ns]'])*last_drift_interval
    
    # Put the array together with all of them
    time_max_chan_intervals = np.concatenate((time_max_chan_intervals,last_time_max_chan_interval))
    
    # Print out the results if desired
    if verbose:
        print(f'Drift Slices: {drift_slices}')
        print(f'Time of Max Channel Slices: {time_max_chan_bins}')
        print(f'Drift Intervals: {drift_intervals}')
        print(f'Time of Max Channel Intervals: {time_max_chan_intervals}')
        
    return drift_time, drift_slices, time_max_chan_bins, drift_intervals, time_max_chan_intervals, iqr_drift_time, dt_start_samples, last_time_max_chan_interval

# fitCentroidstoChargeData()
#
# fitCentroidstoChargeData() is an intermediate function that is called within another intermediate function,
# binAndFitChargeData(), which is itself called within binnedElectronLifetimeFit() to help
# split up the fitting process so as not to make the full fit function extremely long.
# This function takes a histogram with its bin centers and edges, along with an initial guess of
# where the centroid is expected based on the guessed lifetime of the purification data, and 
# it fits a histogram to the data, visualizing many attributes and tracking some statistics
# throughout the process. 
#
# More specifically, this function selects a region within the histogram to fit the Gaussian to, based 
# on the user-guessed lifetime and where the centroid is expected to lie within the data. This 
# sub-region within the histogram is selected to combat noise effects from all the data. The size of
# the subregion is dependent on the interquartile range (IQR) of the data and on the number of events
# histogrammed, with more events corresponding to a narrow range selected.
#
# Many of the inputs and outputs of this function are used for visualization and statistics tracking,
# hence its numerous inputs and outputs. This function is also called within a "for" loop, hence its
# apparently inconsistent variable naming.
#
# NOTE/WARNING: This function is the one that calls scipy.optimize.curve_fit(), and this is the function that
# has some hard-coded values for thresholds based on the statistics from reduced_v9 datasets from
# Run 34. In the future, it would be wise to replace curve_fit() with tools from lmfit, and it would 
# also be wise to determine a more dynamic method of calculating such thresholds that are hard-coded.
# (This message was written at the time of creation, 17 August 2022.)
#
# If curve_fit() fails to converge, then, due to low statistics of the datasets, the mean and std. dev
# of the subregion of the histogram is taken as the centroid and std. dev. thereof for that slice.
#
# Inputs:
#
# bin_centers - a numpy array (float) of all the bin centers of tile energy (ADC Counts) from the 
#               histogram
#
# bin_edges - a numpy array (float) of all the bin edges of tile energy (ADC Counts) from the 
#             histogram (length of array is 1 more than that of bin_centers)
#
# hist - a numpy arary (int) of all the bin counts of tile energy; i.e. the actual histogram
#        for the drift slice
#
# initial_centroid - the initial guess of the centroid (float) based on the user-input
#                    guess of the lifetime
#
# iqr_charge_detected - the interquartile range (float) of all the data points in the 
#                       histogram
#
# drift_slice - the current left edge of the drift slice (float) in µs in which the fit is 
#               being performed (function is called within a for loop)
#
# drift_interval - the current duration of the drift slice (float) in µs in which the fit 
#                  is being performed (function is called within a for loop)
#
# ndx - the current index (int) of the for loop (used ofr plotting)
#
# length_drift_slices - the number of drift slices (int); used for a figure handle
#
# plotFlag - a flag (boolean) that turns plotting on/off; set to "True" by default
#
# verbose - a flag (boolean) that specifies if results are desired to be printed; set to "True" by default
#
# Outputs:
#
# centroid - the centroid (float) of the histogram, i.e. the mean of the Gaussian fit. This point
#            is used as one of the fit points for the actual lifetime fit
#
# centroid_std - the standard deviation of the above value (float)
#
# parameters - a numpy array (float) that has all 3 fit paramters of the Gaussian, in order:
#              Scale factor, mean, std. dev
#
# num_x_fit_bins - the size (int) of the neighborhood within the histogram that was used for 
#                  the Gaussian fit
#
# fit_region_size_ratio - the ratio (float) of the number of bins used for the fit to the number
#                         of bins in the entire histogram
#
# num_events_for_fit - the count (int) of events used in the actual fit
#
# fit_region_event_ratio - the ratio (float) of the number of events used for the fit to the number
#                          of events in the entire histogram
#
# events_to_bin_num_fit_region - the ratio (float) of the number of events used for the fit to
#                                the number of bins used for the fit, i.e. the average number of
#                                events per bin in the fit region
#
# fit_fail_count_flag - a flag (boolean) that determines if the curve_fit() failed
#
# ChangeLog:
# 2022-08-17 MM Created.

def fitCentroidstoChargeData(bin_centers, bin_edges, hist, initial_centroid, iqr_charge_detected, drift_slice, drift_interval, ndx, length_drift_slices, plotFlag=True, verbose=True):
    
    # This array has a list of colors to visualize each slice on a plot of the cut data
    color_name_strings = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    # WARNING: This needs to be more dynamic - the conditional is piecewise and hard-coded
    # These numbers were based on the statistics from reduced_v9 datasets from Run 34
    # Determine the energy subrange within the histogram within which to select the neighborhood to perform the fit;
    # A multiple of the interquartile range is selected around the guessed centroid, and any bin that falls within that range is used 
    # as the fit neighborhood. These hard-coded values are the number by which the IQR is divided when determining the range
    num_events_per_hist = np.sum(hist) # Use the total number of events in the histogram to determine this (there's an apparent correlation)
    if num_events_per_hist < 60:  # For fewer than 60 events histogrammed
        iqr_scale_factor = 1.5    # The IQR is divided by 1.5 on either side of the mean
    elif num_events_per_hist > 60 and num_events_per_hist < 100: # For between 60 - 100 events
        iqr_scale_factor = 1.75   # The IQR is divided by 1.75 on either side of the mean
    else:                         # For greater than 100 events
        iqr_scale_factor = 2      # Half the IQR on either side of the mean is selected (corresponding to 1 full IQR's worth of bins)
        
    # Select the neighborhood:
    # Start with the left index; all bins within 1 scaled IQR below the guessed mean
    left_ndx = np.where(bin_centers>=initial_centroid-iqr_charge_detected/iqr_scale_factor)
    # Print out the results, indices and energy values that apply
    if verbose:
        print(f'left_ndx pre-selection: {left_ndx}')
        print(f'Energies from Left Index: {bin_centers[left_ndx]}')
    # Select the left-most bin to establish the lower bound of the fit neighborhood
    left_ndx = left_ndx[0][0]
    # Print the result
    if verbose:
        print(f'left_ndx post-selection: {left_ndx}')
    # Now the right index; all bins withing 1 scaled IQR above the guessed mean
    right_ndx = np.where(bin_centers<=initial_centroid+iqr_charge_detected/iqr_scale_factor)
    # Print all the indices and values that apply
    if verbose:
        print(f'right_ndx pre-selection: {right_ndx}')
        print(f'Energies from Right Index: {bin_centers[right_ndx]}')
    # Select the right-most bin to establish the upper bound of the fit neighborhood
    right_ndx = right_ndx[0][-1] + 1
    # Print the result
    if verbose:
        print(f'right_ndx post-selection: {right_ndx}')
        
    # Select the actual neighborhood
    x_fit = bin_centers[left_ndx:right_ndx] # From the bin centers (ADC Counts)
    y_fit = hist[left_ndx:right_ndx]        # From the histogram values (counts)
    
    # WARNING: This needs to be more dynamic - the conditional is piecewise and hard-coded
    # These numbers were based on the statistics from reduced_v9 datasets from Run 34
    # This error handle elimiates the right-most bin of the fit neighborhood if it is sufficiently
    # larger than the next-to-right-most bin that curve_fit() fails to converge
    if y_fit[-1]>1.3*y_fit[-2]:                     # If the right-most bin has more than 1.3 times the number of counts of the next-right-most bin
        x_fit = bin_centers[left_ndx:right_ndx-1]   # Remove the right-most bin from the bin centers sub-array
        y_fit = hist[left_ndx:right_ndx-1]          # Remove the right-most bin from the histogram sub-array
    # Report what the values are
    if verbose:
        print(f'Neighborhood of Energy Selection for Gauss Fit: {initial_centroid-iqr_charge_detected/iqr_scale_factor} - {initial_centroid+iqr_charge_detected/iqr_scale_factor} ADC Counts') # The energy range in ADC counts of bins that were selected
        print(f'x_fit: {x_fit}') # The bin centers used for the fit
        print(f'y_fit: {y_fit}') # The hist counts used for the fit
        
    # Get Some Statistics on the Fit Neighborhood
    num_x_fit_bins = np.size(x_fit)                                   # The number of bins used for the Gaussian fit
    fit_region_size_ratio = num_x_fit_bins/np.size(bin_centers)       # Ratio of number of bins used for the fit to total number of bins
    num_events_for_fit = np.sum(y_fit)                                # Number of events for the fit
    fit_region_event_ratio = num_events_for_fit/num_events_per_hist   # Ratio of number of events for the fit to total number of events
    events_to_bin_num_fit_region = num_events_for_fit/num_x_fit_bins  # Average number of events per bin in the fit region
    # Print all results
    if verbose:
        print(f'Number of Bins Used for Gaussian Fit: {num_x_fit_bins}')
        print(f'Ratio of Fit Bins to Total Hist Bins: {fit_region_size_ratio}')
        print(f'Number of Events Used in Fit: {num_events_for_fit}')
        print(f'Ratio of Events Used in Fit to Total Events in Hist: {fit_region_event_ratio}')
        print(f'Events to Bin Number Ratio in Fit Region: {events_to_bin_num_fit_region}')
        
    # Add the X-Fit Region to the Graph with the Histogram Boxes; add vertical lines to the fit plots
    if plotFlag:
        plt.figure(length_drift_slices+2)        # This is the second figure generated after each of the histogram fits are done
        x_fit_height = x_fit[-1] - x_fit[0]    # This is the height of the rectangle that will be added to this figure
        # This rectangle is added to the plot of the cut data used for fitting; it shows the energy range used for the fit
        rect_x_fit = mpatches.Rectangle((drift_slice,x_fit[0]),drift_interval,\
                            x_fit_height,fill = False,color=color_name_strings[ndx%len(color_name_strings)],\
                            linewidth=2,\
                            label=f'Fit Neighborhood: ${x_fit[0]:.2f} - ${x_fit[-1]:.2f} ADC Counts', linestyle='--')
        plt.gca().add_patch(rect_x_fit)        # Adds the rectangle
        plt.legend(bbox_to_anchor=(1.1,1.0))   # Moves the legend for visibility
        # This section adds vertical black bars at the lower and upper bounds of the fit region on the histogram for this slice
        right_bar = bin_edges[np.where(bin_edges>x_fit[-1])]    # The upper bound, all energies greater than the upper bound
        right_bar = right_bar[0]                                # The first of these energies
        left_bar = bin_edges[np.where(bin_edges<x_fit[0])]      # The lower bound, all energies lower than the lower bound
        left_bar = left_bar[-1]                                 # The last of these energies
        plt.figure(ndx)    # Reference the figure with the individual fit for this slice
        plt.axvline(x = left_bar,color='black',linewidth=2,label='Left Bound Used for Fit')    # Plot the lower bound vertical line
        plt.axvline(x = right_bar,color='black',linewidth=2,label='Right Bound Used for Fit')  # Plot the upper bound vertical line
    
    # Make the Fit; if it Fails, just use the mean/std of the hist region
    fit_fail_count_flag = False       # This flag is sent to the parent function to increase the count on all failed fits
    try:
        parameters, covariance = gauss_fit(x_fit,y_fit)   # This is the actual fit; lmfit should be switched to in the future
    except:                           # If the fit fails:
        # Alert the user
        if verbose:
            print('Fit Failed: Using Standard Statistics of Hist Data')
        parameters = np.array([0, 0, 0],dtype=float)    # Preallocate the space for the array in the way curve_fit() returns these values
        parameters[0] = np.amax(y_fit)                  # First value is the scale factor (not used)
        parameters[1] = np.mean(x_fit)                  # Second value is the mean (i.e. centroid), just the mean of the region's energies
        parameters[2] = np.std(x_fit)                   # Third value is the std. dev, just the std. dev of the region's energies
        fit_fail_count_flag = True                      # Set the flag to True to increase the count of failed fits in the parent function
    centroid = parameters[1]                            # Assign the centroid value (parameters[1] is used for plots)
    centroid_std = parameters[2]                        # Assign the centroid std. dev. value (parameters[2] is used for plots)
    # Report the centroid with uncertainty
    if verbose:
        print(f'Centroid: {centroid} \u00B1 {centroid_std} ADC Counts')
    
    # Return everything
    return centroid, centroid_std, parameters, num_x_fit_bins, fit_region_size_ratio, num_events_for_fit, fit_region_event_ratio, events_to_bin_num_fit_region, fit_fail_count_flag

# binAndFitChargeData()
#
# binAndFitChargeData() primarily bins all the charge data, then calls fitCentroidstoChargeData()
# to fit the data with Gaussians for each drift slice. A "for" loop is called to bin and fit the 
# data for each drift slice, the slices and thicknesses calculated by cutDriftSlices(). 
#
# Many of the inputs and outputs of this function are for visualization and statistics-recording
# purposes, hence the many inputs.
#
# Quality cuts are applied to charge tile data vs. drift time, including linear cuts on the tile 
# energy as a function of drift time, and from the cut data, the number of hist bins for each slice
# is calculated using twice the suggested value by the Freedman-Diaconis rule. The data is then 
# histogrammed, and the bin centers are formed from the bin edges. The fit is then carried out in
# fitCentroidstoChargeData(). 
#
# Inputs: 
# 
# data_df - the pandas dataframe object (hdf5) that contains all information on the dataset from the run
#
# drift_time - the extracted numpy array (float) of the drift time in µs of all events in the dataset
#
# charge_energy - the extracted numpy array (float) of the tile enery in ADC counts of all events in the dataset
# 
# initial_centroids - the numpy array (float) of centroid guesses as predicted by the user-input lifetime and 
#                     intercept guesses for the dataset
# 
# drift_slices - the numpy array (float) of left edges of the drift time slices in µs that divides the data
#                for fitting
#  
# drift_intervals - the numpy array (float) of slice thicknesses of above
# 
# time_max_chan_bins - the numpy array (int?) of left edges of the drift time slices in samples 
#                      that divides the data for fitting
# 
# time_max_chan_intervals - the numpy array (int?) of slice thicknesses of above
# 
# sipm_lower_bound - the lower boundary (float) of the SiPM energy used in quality cuts to visualize the alpha
#                    band; user-input
# 
# sipm_upper_bound - the upper boundary (float) of the SiPM energy used in quality cuts to visualize the alpha
#                    band; user-input
# 
# m_cuts - the slope (float) of the linear cuts on tile energy applied to the data before histogramming
# 
# b_low - the lower-bound intercept (float) of the linear cuts on tile energy applied to the data before 
#         histogramming
# 
# b_high - the upper-bound intercept (float) of the linear cuts on tile energy applied to the data before 
#         histogramming
#
# run_num - the name of the run (string) for saving/titling figures
#
# this_dataset - the number of the dataset (string) for saving/titling figures
# 
# plotFlag - a flag (boolean) that turns plotting on/off; set to "True" by default
#
# verbose - a flag (boolean) that specifies if results are desired to be printed; set to "True" by default
# 
# Outputs: 
#
# centroids - a numpy array (float) of all the means of the Gaussian fits for each drift slice
#
# centroids_stds - a numpy array (float) of all the std. devs of the Gaussian fits for each drift slice
#
# hist_bin_numbers - a numpy array (int) of the total number of histogram bins used in each slice
#  
# num_events_per_hist - a numpy array (int) of the number of events in each histogram per drift slice 
# 
# event_to_bin_num_ratio - a numpy array (float) of the average number of events per bin in each hist
#
# num_x_fit_bins - a numpy array (int) of the number of bins actually used to fit the Gaussian within
#                  the histogram for each drift slice
# 
# num_events_for_fit - a numpy array (int) of the number of events actually used to fit the Gaussian 
#                      within the histogram for each drift slice
# 
# events_to_bin_num_fit_region - a numpy array (float) of the average number of events per bin in the 
#                                bins used for the Gaussian fit for each drift slice
# 
# fit_region_size_ratio - a numpy array (float) of the ratio of number of bins used for the Gaussian
#                         fit to the number of bins in the histogram for each drift slice
#
# fit_region_event_ratio - a numpy array (float) of the ratio of number of events used for the Gaussian
#                         fit to the number of events in the histogram for each drift slice
# 
# fit_fail_count - the count (int) of the number of drift slices who had their fits fail by curve_fit()
# 
# fig_fname - the file name (string) of the path down which the figure gets saved
# 
# ChangeLog:
# 2022-08-17 MM Created.

def binAndFitChargeData(data_df, drift_time, charge_energy, initial_centroids, drift_slices, drift_intervals, time_max_chan_bins, time_max_chan_intervals, sipm_lower_bound, sipm_upper_bound, m_cuts, b_low, b_high, run_num, this_dataset, plotFlag=True, verbose=True):
    
    # This array has colors to help visualize the regions and slices that are histogrammed on the cut data plot
    color_name_strings = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    # Preallocate space for fit values and statistics
    centroids = np.zeros(len(drift_slices))                      # The means of the Gaussian fits (used for the lifetime fit)
    centroids_stds = np.zeros(len(drift_slices))                 # The std. devs of the Gaussian fits
    hist_bin_numbers = np.zeros(len(drift_slices))               # The number of bins used in each histogram per drift slice
    num_events_per_hist = np.zeros(len(drift_slices))            # The number of events in each histogram
    event_to_bin_num_ratio = np.zeros(len(drift_slices))         # The average number of events per bin in each histogram
    num_x_fit_bins = np.zeros(len(drift_slices))                 # The number of bins actually used in the fit per drift slice
    num_events_for_fit = np.zeros(len(drift_slices))             # The number of events actually used in the fit per drift slice
    events_to_bin_num_fit_region = np.zeros(len(drift_slices))   # The average number of events per bin in the fit region per drift slice
    fit_region_size_ratio = np.zeros(len(drift_slices))          # The ratio of number of fit region bins to total number of hist bins per drift slice
    fit_region_event_ratio = np.zeros(len(drift_slices))         # The ratio of number of fit region events to total number of events histogrammed per drift slice 
    fit_fail_count = 0                                           # The count of slices that had their fits fail
    length_drift_slices = np.size(drift_slices)                  # This length is used for a figure reference in fitCentroidstoChargeData()
    
    # Loop over each drift slice to perform the binning and fitting
    for ndx in range(len(drift_slices)):
        # Print the drift slice, start to end, in µs and in samples
        if verbose:
            print('Drift Slice '+str(ndx)+f': {drift_slices[ndx]:.2f} - {(drift_slices[ndx]+drift_intervals[ndx]):.2f} µs')
            print('Drift Slice '+str(ndx)+f': {time_max_chan_bins[ndx]:.2f} - {(time_max_chan_bins[ndx]+time_max_chan_intervals[ndx]):.2f} Samples')
        # Now actually Cut the Data into Drift Time Intervals
        dt_mask= (data_df['TimeOfMaxChannel'] > time_max_chan_bins[ndx]) & \
            (data_df['TimeOfMaxChannel'] < (time_max_chan_bins[ndx]+time_max_chan_intervals[ndx]))\
            & (data_df['TotalSiPMEnergy']> sipm_lower_bound) & (data_df['TotalSiPMEnergy']< sipm_upper_bound) \
            & (data_df['NumTileChannelsHit'] < 3) & (data_df['IsFull3D']) \
            & (data_df['TotalTileEnergy']> m_cuts*drift_time+b_low) &\
            (data_df['TotalTileEnergy']< m_cuts*drift_time+b_high)     # Linear cuts in tile energy as a function of drift time are applied, square cuts in SiPM energy are applied
        # Apply the Cuts to the Tile Energy for Histogramming
        charge_detected=charge_energy[dt_mask]   
        bin_range_edge_left = np.amin(charge_detected)  # The minimum value of energies to be histogrammed in this slice
        bin_range_edge_right = np.amax(charge_detected) # The maximum value of energies to be histogrammed in this slice
        # Add rectangles to the Cut Data Plot, Also the Centroid Guess as a Dashed Line
        if plotFlag:
            plt.figure(len(drift_slices)+2)             # This figure is a plot of the cut data actually used for histogramming
            plt.plot(drift_time[dt_mask],charge_energy[dt_mask],'o',color=(0.,0.,1.,0.5),markersize=5.,\
                 markeredgecolor=(0.,0.,0.,0.))         # Plot all the data
            plt.xlim([0,np.amax(drift_time[dt_mask])])  # Set the x axis limit to 0 µs
            plt.ylim([0,b_high])                        # Set the y axis limit to 0 ADC Counts
            # Add x, y labels, title
            plt.xlabel("Drift time (µs)")                     
            plt.ylabel("Charge Tile Energy (ADC Counts)")
            plt.title("Data Used for Histogramming/Fits with Drift Slices and Centroid Guesses, " + run_num + ", " + this_dataset)
            charge_height = bin_range_edge_right - bin_range_edge_left # This is the height of the rectangle that visualizes the drift slice
            # This is the rectangle that shows the drift slice, its thickness, and the range of the data that is histogrammed within it
            rect = mpatches.Rectangle((drift_slices[ndx],bin_range_edge_left),drift_intervals[ndx],\
                               charge_height,fill = False,color=color_name_strings[ndx%len(color_name_strings)],\
                              linewidth=2,\
                              label=f'${drift_slices[ndx]:.2f} - ${drift_slices[ndx]+drift_intervals[ndx]:.2f} µs')
            plt.gca().add_patch(rect) # Add the rectangle to the plot
            # This does not show up as a rectangle, but as a line that shows where the guessed centroid lies within the drift slice
            rect_centroid = mpatches.Rectangle((drift_slices[ndx],initial_centroids[ndx]),drift_intervals[ndx],\
                               0,fill = False,color=color_name_strings[ndx%len(color_name_strings)],\
                              linewidth=2,\
                              label=f'Initial Guess: {initial_centroids[ndx]:.2f} ADC Counts', linestyle="-.")
            plt.gca().add_patch(rect_centroid) # Add the line
            plt.legend(bbox_to_anchor=(1.1,1.0))  # Move the legend to be able to see the plot
        
        iqr_charge_detected = stat.iqr(charge_detected,rng=(25, 75),scale=1.0,nan_policy="omit") # Use the IQR for later, with selecting the size of the region within the histogram for the fit
        hist_bin_num = 2*freedmanDiaconis(charge_detected,returnas="eat my asshole") # Determine the number of bins by which to histogram the data - Twice the F-D suggestion has best resolution for a Gaussian fit given the energy range of binning
        hist_bin_numbers[ndx] = hist_bin_num # Store this value 
        # Report some values 
        if verbose:    
            print(f'Number of Events in This Slice to Be Histogrammed: {np.size(charge_energy[dt_mask])}')
            print(f'IQR of The Charge: {iqr_charge_detected}')
            print(f'Number of Charge Histogram Bins: {hist_bin_num}')
        # Histogram the Data in the Drift Slice
        hist, bin_edges = np.histogram(charge_detected,bins=hist_bin_num,\
                                       range=(bin_range_edge_left,bin_range_edge_right))
        num_events_per_hist[ndx] = np.sum(hist)   # Tally/store the total events in the histogram
        event_to_bin_num_ratio[ndx] = num_events_per_hist[ndx]/hist_bin_num # Calculate the average number of events per bin for this histogram
        # Print these values/arrays
        if verbose:
            print(f'Counts per bin: {hist}')
            print(f'Bin Edges: {bin_edges}')
            print(f"Number of Bins used for Histogramming: {hist_bin_num}")
            print(f'Number of Events Per Histogram: {num_events_per_hist[ndx]}')
            print(f'Event/Bin Num Ratio of Histogram: {event_to_bin_num_ratio[ndx]}')
        # Make Bin Centers from Bin Edges Array, Then Plot the Histogram that way
        bin_centers = np.array([(bin_edges[x+1]+bin_edges[x])/2 for x in range(len(bin_edges)-1)])
        # Print the Bin Centers
        if verbose:
            print(f'Bin Centers: {bin_centers}')
        if plotFlag:
            plt.figure(ndx)       # One Figure per Drift Slice, which contains a Histogram of the data, a vertical dashed black line showing the centroid guess, two solid black lines showing the bins that were used for the fit, a red curve showing the Gaussian fit, and a vertical red line showing the fit centroid
            plt.hist(charge_detected,bins=hist_bin_num,range=(bin_range_edge_left,bin_range_edge_right),\
                     label=f'Drift Time Interval: {drift_slices[ndx]:.2f} - {(drift_slices[ndx]+drift_intervals[ndx]):.2f} µs ({num_events_per_hist[ndx]} Events)')   # Plot the histogram
#             plt.xlim([0,np.amax(bin_edges)])
            plt.axvline(x = initial_centroids[ndx],color='black',linestyle='--',\
                        linewidth=1,label='Guessed Centroid Based on Lifetime Guess')   # Plot the guessed centroid
            # Label titles, axes, legend
            plt.title('Drift Slice '+str(ndx)+f': {drift_slices[ndx]:.2f} - {(drift_slices[ndx]+drift_intervals[ndx]):.2f} µs')
            plt.xlabel('Total Tile Energy (ADC Counts)')
            plt.ylabel('Frequency')
            plt.legend(loc='best')
            # Set the name of the figure for it to be saved
            fig_fname = 'Gauss_Fit_'+run_num+'_'+this_dataset+f'_{drift_slices[ndx]:.2f}_{(drift_slices[ndx]+drift_intervals[ndx]):.2f}_us.png'
        # Determine Neighborhood for Fit
        if verbose:
            print(f'Initial Centroid Guess: {initial_centroids[ndx]}')
        # Fit the Neighborhood - See fitCentroidstoChargeData() for details
        centroid_ndx, centroid_std_ndx, parameters, num_x_fit_bins_tmp, fit_region_size_ratio_tmp, num_events_for_fit_tmp, fit_region_event_ratio_tmp, events_to_bin_num_fit_region_tmp, fit_fail_flag = fitCentroidstoChargeData(bin_centers, bin_edges, hist, initial_centroids[ndx], iqr_charge_detected, drift_slices[ndx], drift_intervals[ndx], ndx, length_drift_slices, plotFlag, verbose)
        # Increase the fail count if the fit was unsuccessful with curve_fit()
        if fit_fail_flag:
            fit_fail_count = fit_fail_count + 1
        # Store values from the fit function for this drift slice:
        centroids[ndx] = centroid_ndx                                              # The fitted centroid
        centroids_stds[ndx] = centroid_std_ndx                                     # The std dev of the fitted centroid
        num_x_fit_bins[ndx] = num_x_fit_bins_tmp                                   # The number of bins used for the fit
        fit_region_size_ratio[ndx] = fit_region_size_ratio_tmp                     # The ratio of the number of bins used for the fit to total number of bins 
        num_events_for_fit[ndx] = num_events_for_fit_tmp                           # The number of events used for the fit
        fit_region_event_ratio[ndx] = fit_region_event_ratio_tmp                   # The ratio of number of events used for the fit to total number of events
        events_to_bin_num_fit_region[ndx] = events_to_bin_num_fit_region_tmp       # The average number of events per bin in the bins used for the fit
        # Add the Gaussian and fit centroid to the histogram plot
        if plotFlag:
            plt.figure(ndx)      # Reference the proper figure
            x_fit_plot = np.linspace(bin_centers[0],bin_centers[-1],1000)   # A smooth line to make things look nice
            plt.plot(x_fit_plot, gauss(x_fit_plot,parameters[0],parameters[1],parameters[2]), color="red",\
                label=f'y = {parameters[0]:.2f}*exp(-(x-{parameters[1]:.2f})^2/(2*{parameters[2]:.2f}^2))') # Add the Gaussian curve
            plt.axvline(x = centroids[ndx],color='red',linewidth=2,label='Mean calculated by Gaussian Fit') # Add the centroid line
            plt.legend(bbox_to_anchor=(1.1,1.0)) # Set the legend location out of the way of things
            plt.savefig(fig_fname,format='png')  # Save the figure
        
        # Skip a line in the printing to make reading things more digestible
        if verbose:
            print('')
            
    # Return everything
    return centroids, centroids_stds, hist_bin_numbers, num_events_per_hist, event_to_bin_num_ratio, num_x_fit_bins, num_events_for_fit, events_to_bin_num_fit_region, fit_region_size_ratio, fit_region_event_ratio, fit_fail_count, fig_fname

# binnedElectronLifetimeFit()
#
# binnedElectronLifetimeFit() is the main function of this class, the function to be called when
# estimating the lifetime of a purification dataset. It calls cutDriftSlices(), binAndFitChargeData()
# (which itself calls fitCentroidstoChargeData()), and lsLin() to complete this task.
#
# The function passes inputs into cutDriftSlices(), which creates the drift slices in which
# centroids are fit for the actual lifetime fit. The slices are used, then, to calculate the 
# intercept guesses, which are calculated from the user-guessed lifetime and intercept they
# believe the data has. Based on the statistics of the data used to determine the number and 
# sizes (non-uniform) of the drift slices, linear cuts on the tile energy are established for
# the data, which is then passed into binAndFitChargeData() to histogram each slice by tile
# energy and fit those plots to a Gaussian. The centroids (means) of these fits are then passed
# into lsLin(), which fits the natural log of these points to a linear fit, the slope of which
# is the opposite of the reciprocal of the lifetime. 
# 
# Relevant statistics and plots are presented at the discretion of the user.
#
# Many of the current issues (at the time of writing, 17 August 2022) and improvements to this
# function actually exist in the modular functions called by this one; however, switching to 
# lmfit instead of scipy.optimize.curve_fit() and handling the uncertainties of the centroids
# when performing the lifetime fit, also reporting the reduced chi-squared for the final fit and
# for each Gaussian fit would be a major professional improvement.
#
# Inputs:
#
# intercept_guess - the user-specified guess (float) of where the "intercept" of the alpha band
#                   is located on the plot of the data's tile energy vs. drift time (in ADC Counts)
#
# lifetime_guess - the user-specified guess (float) of the electron lifetime of the purification
#                  data
#
# start_drift - the user-specified lower bound (float) of drift time (in µs) where the fit is desired to
#               take place (start at at least 10 µs due to field effects in the chamber)
# 
# end_drift - the user-specified upper bound (float) of drift time (in µs) where the fit is desired
#             to take place (right before where the data gets particularly noisy toward the bottom
#             of the chamber)
#
# data_df - the pandas dataframe (hdf5) object that contains all information for all events in the
#           dataset
#
# analysis_config - the configuration object that contains information about the run parameters of 
#                   the dataset
#
# sipm_lower_bound - the user-specified lower boundary (float) of SiPM energy of all events in 
#                    ADC counts
#
# sipm_upper_bound - the user-specified upper boundary (float) of SiPM energy of all events in 
#                    ADC counts
#
# charge_lower_bound - the user-specified lower boundary (float) of charge tile energy of all events 
#                      in ADC counts
#
# charge_upper_bound - the user-specified lower boundary (float) of charge tile energy of all events 
#                      in ADC counts 
#
# run_num - the name of the run (string) for saving/titling figures
#
# this_dataset - the number of the dataset (string) for saving/titling figures
#
# plotFlag - a flag (boolean) that turns plotting on/off; set to "True" by default
#
# verbose - a flag (boolean) that specifies if results are desired to be printed; set to "True" by default
#
# Outputs:
#
# lifetime - the calculated electron lifetime (float) of the dataset, in µs, from the least-squares
#            fit
# 
# sigma_lifetime - the calculated uncertainty (float) of the above, in µs
# 
# intercept - the calculated "intercept" of the alpha band located on the plot of
#             the data's tile energy vs. drift time, in ADC counts
#
# sigma_intercept - the calculated uncertainty (float) of the above, in ADC counts
#
# r_squared - the coefficient of determination (float) of the linear fit used to calculate the lifetime
#
# drift_slices - the numpy array (float) of left edges of the drift time slices in µs that divides the data
#                for fitting
#
# centroids - a numpy array (float) of all the means of the Gaussian fits for each drift slice
#
# centroids_stds - a numpy array (float) of all the std. devs of the Gaussian fits for each drift slice 
#
# ChangeLog:
# 2022-08-17 MM Created.

def binnedElectronLifetimeFit(intercept_guess,lifetime_guess,start_drift,end_drift,data_df, analysis_config,sipm_lower_bound, sipm_upper_bound,charge_lower_bound,charge_upper_bound,run_num,this_dataset,plotFlag=True,verbose=True):
    # Extract arrays from the dataframe file
    charge_energy = data_df['TotalTileEnergy']         # The tile energy, the dependent variable in the main plot of the data
    time_of_max_channel = data_df['TimeOfMaxChannel']  # The drift time, in samples, the independent variable in the main plot of the data
    
    # Get Drift Time and Sample Slices and Relevant Stamps for Plotting
    drift_time, drift_slices, time_max_chan_bins, drift_intervals,\
        time_max_chan_intervals, iqr_drift_time, dt_start_samples,\
           last_time_max_chan_interval = cutDriftSlices(data_df, charge_energy,\
                                                                time_of_max_channel, analysis_config, start_drift,\
                                                                end_drift, sipm_lower_bound, sipm_upper_bound,\
                                                                charge_lower_bound, charge_upper_bound, verbose)
    
    # Create Centroid Guesses From Guessed Lifetime + Drift Time Bins (the lifetime follows an exponential decay)
    initial_centroids = intercept_guess*np.exp(-drift_slices/lifetime_guess)
    
    # Determine Lines that Parameterize the Linear Quality Cuts on Tile Energy
    intercept_offset = 2*iqr_drift_time       # The linear cuts span one full interquartile range on either side of the guessed intercept
    b_low = intercept_guess - intercept_offset   # Set the intercept for the lower linear cut on tile energy
    b_high = intercept_guess + intercept_offset  # Set the intercept for the upper linear cut on tile energy
    m_cuts = (initial_centroids[-1]-intercept_guess)/(end_drift - start_drift)  # Calculate the slope based on the "endpoints" of the data
    # Report these cuts values
    if verbose:
        print(f'Intercept of Charge Energy Lower Cut Line: {b_low}')
        print(f'Intercept of Charge Energy Upper Cut Line: {b_high}')
        print(f'Slope of Charge Energy Cut Lines: {m_cuts}')
    
    # Visualize the linear cuts on the data
    if plotFlag:
        mask_check = (data_df['TotalTileEnergy']> m_cuts*drift_time+b_low) & \
            (data_df['TotalTileEnergy']< m_cuts*drift_time+b_high)\
            & (data_df['TotalSiPMEnergy']>sipm_lower_bound) & (data_df['TotalSiPMEnergy']<sipm_upper_bound) \
            & (data_df['NumTileChannelsHit'] < 3) & (data_df['IsFull3D'])\
            & (data_df['TimeOfMaxChannel']> dt_start_samples) & (data_df['TimeOfMaxChannel']< float(time_max_chan_bins[-1]+last_time_max_chan_interval))  # Create the proper mask for the data
        plt.figure(len(drift_slices)+1)                     # This figure will show all the cut data that is actually used for binning/fitting
        plt.plot(drift_time[mask_check],charge_energy[mask_check],'o',color=(0.,0.,1.,0.5),markersize=5.,\
             markeredgecolor=(0.,0.,0.,0.)) # Plot the data
        plt.xlim([0,np.amax(drift_time[mask_check])])     # Set the drift time limit to 0 µs
        plt.ylim([0,np.amax(charge_energy[mask_check])])  # Set the tile energy limit to 0 ADC Counts
        # Add labels and titles
        plt.xlabel("Drift time (µs)")
        plt.ylabel("Charge Tile Energy (ADC Counts)")
        plt.title("Data that is Histogrammed/Fit For Lifetime, " + run_num + ", " + this_dataset)
    if verbose:
        print('')  # Skip a line for readability 
        
    # Histogram/Fit Each Drift Slice
    centroids, centroids_stds, hist_bin_numbers, num_events_per_hist, event_to_bin_num_ratio,\
        num_x_fit_bins, num_events_for_fit, events_to_bin_num_fit_region, fit_region_size_ratio,\
        fit_region_event_ratio, fit_fail_count, fig_fname = binAndFitChargeData(data_df, drift_time, charge_energy, initial_centroids, drift_slices, drift_intervals, time_max_chan_bins,\
                        time_max_chan_intervals, sipm_lower_bound, sipm_upper_bound, m_cuts, b_low, b_high,\
                       run_num, this_dataset, plotFlag, verbose)
    # Report these values
    if verbose:        
        print('All Drift Slice Data:')
        print(f'Centroids Determined by Fit: {centroids}')                                                    # All fit points
        print(f'Std. Devs. of Centroids Determined by Fit: {centroids_stds}')                                 # All uncertainties
        print(f'{fit_fail_count} out of {len(drift_slices)} Fits Failed')                                     # Number of failed fits
        print(f'Number of Histogram Bins per Slice: {hist_bin_numbers}')                                      # All totals of hist bins
        print(f"Number of Events per Drift Time Slice: {num_events_per_hist}")                                # All totals of events/hist
        print(f"Ratios of Number of Events per Drift Time Bin to Number of Bins: {event_to_bin_num_ratio}")   # All avgs. of events/bin
        print(f'Number of Bins Used for Gaussian Fit: {num_x_fit_bins}')                                      # All totals of fit bins
        print(f'Ratio of Fit Bins to Total Hist Bins: {fit_region_size_ratio}')                               # All ratios of fit bin counts to hist bin counts
        print(f'Number of Events Used in Fit: {num_events_for_fit}')                                          # All totals of events/fit
        print(f'Ratio of Events Used in Fit to Total Events in Hist: {fit_region_event_ratio}')               # All ratios of fit event counts to hist event counts
        print(f'Events to Bin Number Ratio in Fit Region: {events_to_bin_num_fit_region}')                    # All avgs. of events/fit bin
    
    # Plot the fitted centroids with errorbars with no data, just to see them by themselves
    if plotFlag:
        plt.figure(len(drift_slices)+3)           # This figure contains only the centroids with uncertainties
        plt.errorbar(drift_slices,centroids,fmt='o--',yerr=centroids_stds)    # Plot with errorbars
        # Add labels to title, axes
        plt.title('Centroids of Binned Tile Energies vs. Drift Time, no Data, '+run_num+', '+this_dataset)
        plt.ylabel('Charge Energy (ADC Counts)')
        plt.xlabel('Drift Time (µs)')
        # Annotate each point with its calculated value (not including errorbars)
        for x, y in zip(drift_slices,centroids):
            label = f'{y:.2f}'      # Generate the label
            plt.annotate(label,(x,y),textcoords="offset points",xytext = (0,10),ha="center") # Annotate with the label
        plt.show() # Add the changes  
        fig_name_2 = 'CentroidsVsDriftTimeNoData_'+run_num+'_'+this_dataset+'.png' # Set the file name for this figure
        plt.savefig(fig_fname,format='png') # Save this figure
        
    # Fit the log of the Centroids to a Least Squares Linear Fit
    ls_slope, ls_sigma_slope, ls_intercept, ls_sigma_intercept, r_squared, N = lsLin(drift_slices,np.log(centroids))
    # Calculate Lifetime
    lifetime = -1/ls_slope                                # Lifetime is the opposite of the reciprocal of the slope
    sigma_lifetime = -(lifetime/ls_slope)*ls_sigma_slope  # Error propagation of the lifetime uncertainty given how the slope is calculated
    intercept = np.exp(ls_intercept)                      # Convert the intercept back to linear scale
    sigma_intercept = ls_sigma_intercept*intercept        # Error propagation of the intercept uncertainty given how the intercept is calculated
    # Report Results of lsLin()
    if verbose:
        print(str(N) +" Drift Slices")                                            # Number of fit points
        print(f"Slope: {ls_slope} \u00B1 {ls_sigma_slope} (MHz, technically)")    # Slope of Least-Squares linear fit
        print(f"Intercept: {ls_intercept} \u00B1 {ls_sigma_intercept}")           # Intercept of Least-Squares linear fit
        
    # Report Final Results
    print(f"Lifetime: {lifetime} \u00B1 {sigma_lifetime} µs")                     # Lifetime with uncertainty
    print(f"R^2 of Log Fit of Centroids: {r_squared}")                            # Quality of linear least squares fit
    
    # Plot the least squares fit on its own, with errorbars
    if plotFlag: 
        mean_x = np.mean(drift_slices)        # x-coordinate of faint points that carries labels for the legend
        mean_y = np.mean(np.log(centroids))   # y-coordinate of faint points that carries labels for the legend
        plt.figure(len(drift_slices)+4)       # This graph shows just the least-squares linear fit of the log of the centroids
        plt.errorbar(drift_slices,np.log(centroids),yerr=centroids_stds/centroids,fmt='o',color=(0.,0.,1.,0.5),markersize=5.,\
             markeredgecolor=(0.,0.,0.,0.),label='Data (Bins are Left Edges)') # Plot the centroids with errorbars
        plt.plot(drift_slices,ls_slope*drift_slices+ls_intercept,'-',\
             color=(0.,0.,0.,1),linewidth=1.5,markeredgecolor=(0.,0.,0.,0.),\
             label=f'Fit: y = (${ls_slope:.4f}  \u00B1  ${ls_sigma_slope:.4f})*x + (${ls_intercept:.2f}  \u00B1  ${ls_sigma_intercept:.2f})')   # Plot the line with the fit parameters 
        plt.plot(mean_x,mean_y,'o',color=(0.,0.,0.,0.01),markersize=0.01,markeredgecolor=(0.,0.,0.,0.),\
             label=f'Electron Lifetime: ${lifetime:.2f}  \u00B1  ${sigma_lifetime:.2f} µs')  # Add a faint point for the lifetime label
        plt.plot(mean_x,mean_y,'o',color=(0.,0.,0.,0.01),markersize=0.01,markeredgecolor=(0.,0.,0.,0.),\
             label=f'R^2: {r_squared:.4f}')         # Add a faint point for the R^2 label
        # Label title, axes, add a legend
        plt.xlabel("Drift time (µs)")
        plt.ylabel("Log Charge Energy (ln(ADC Counts))")
        plt.title("Charge Energy vs. Drift Time, Semilog Fit, Run 34, "+this_dataset+" ("+str(N)+" Drift Slices)")
        plt.legend(loc='best')
        fig_fname3 = 'ChargeEnergyVsDriftTimeSemiLogFit_'+run_num+'_'+this_dataset+'.png' # Create the file name for the figure to be saved
        plt.savefig(fig_fname,format='png') # Save the figure
        
    # Return Everything
    return lifetime, sigma_lifetime, intercept, sigma_intercept, r_squared, drift_slices, centroids, centroids_stds
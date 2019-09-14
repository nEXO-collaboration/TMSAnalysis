import os
import sys
import time
import ROOT

from tms_utilities.digi_filter import digi_hp
from tms_utilities.digi_filter import digi_lp
from tms_utilities.useful_function_shapes import fitShapedConst

import numpy as np
import scipy.optimize as opt
import pandas as pd

######################################################################################
def ParseTMSXwireWfm( raw_waveform, save_waveform = False ):


    output_series = pd.Series()

    # Denoising/filtering happens here.
    channel_fft = np.fft.rfft( raw_waveform )
    channel_fft[100:]=0
    channel_wfm_fft=np.fft.irfft(channel_fft)
    channel_wfm_fft_hp=digi_hp(channel_wfm_fft)
    channel_wfm_fft_hp_lp=digi_lp(channel_wfm_fft_hp)

    graph_fftcut=ROOT.TGraph()
    for isamp in xrange( len(raw_waveform) ):
        x = isamp
        #graph_fftcut.SetPoint(isamp,x,channel_wfm_lp[isamp])
        graph_fftcut.SetPoint(isamp,x,channel_wfm_fft_hp_lp[isamp])

    noise_wfm = np.array([graph_fftcut.GetY()[isamp] for isamp in xrange(0,200,1)])
    signal_wfm = np.array([graph_fftcut.GetY()[isamp] for isamp in xrange(200,800,1)])

    noise_baseline = np.mean(noise_wfm)
    noise_rms=np.sqrt(noise_wfm.dot(noise_wfm)/noise_wfm.size)
           
    signal_amp = np.max(signal_wfm)

    # fit_init is a tuple 
    fit_init_list = list()
    fit_init_list.append( 1. )             # integral
    fit_init_list.append( float(300.0) )  # starting time
    #fit_init_list.append( float(125.*1) )  # differential time
    #fit_init_list.append( float(31.25*2))  # integration time
    fit_init_list.append( noise_baseline)  # noise floor

    if (signal_amp-noise_baseline)>4*noise_rms and (np.argmax(signal_wfm)+300)>300 :

         #fitFunc.SetParameter(0,np.max(signal_wfm)*100)

         fit_init_list[0] = np.max(signal_wfm)*8.
         fit_init = tuple( fit_init_list )
         #print('Init: [{},{},{}]'.format(fit_init_list[0],\
         #                                fit_init_list[1],\
         #                                fit_init_list[2]))

         fit_start = 200#np.argmax(signal_wfm)+2000-150
         fit_end   = 900#np.argmax(signal_wfm)+2000+400

         fit_mask = range(fit_start,fit_end+1)
         
         fit_x = np.arange(float(fit_start),float(fit_end+1))
         fit_y = channel_wfm_fft_hp_lp[fit_mask]  
#         print('fit_x: {}'.format(fit_x))
#         print('fit_y: {}'.format(fit_y))

         try:
           popt, pcov = opt.curve_fit( fitShapedConst, \
                                       fit_x, \
                                       fit_y, \
                                       p0=fit_init_list )
         except RuntimeError as e:
           print(e)
           popt = np.zeros(len(fit_init))
           pcov = np.zeros(len(fit_init))
         #print('Fit: [{},{},{}]'.format(popt[0],popt[1],popt[2]))
#         print('Popt {}'.format(popt))
#         print('Pcov {}'.format(pcov))
#         print('pcov.shape {}'.format(pcov.shape))
#         if pcov.shape
         try:
           perr = np.sqrt( np.diag( pcov ) )
         except ValueError:
           perr = np.zeros(len(popt))


         y_from_fit = fitShapedConst( fit_x, popt[0], popt[1], popt[2] )
         redChi2 = np.sum( (y_from_fit - fit_y)**2 ) / ( len(fit_x) - len(popt) )

         deltaT=0.1*(popt[2]-np.min(y_from_fit))*60
         startT= popt[1]

         if(startT>300) and (startT-300)*0.052<20 and redChi2 <5 :
                output_series['Drift'] = (startT-300)*0.052
                output_series['Energy'] = signal_amp
                output_series['Drift Err'] = perr[1]*0.052
                output_series['Fit RedChi2'] = redChi2
         else: 
                output_series['Drift'] = -1.
                output_series['Energy'] = -1.
                output_series['Drift Err'] = -1.
                output_series['Fit RedChi2'] = redChi2

    else:
         output_series['Drift'] = -1.
         output_series['Drift Err'] = -1.
         output_series['Energy'] = -1.
         output_series['Fit RedChi2'] = -1.

    if save_waveform:
       output_series['TMS Data'] = raw_waveform
       output_series['TMS Denoised Data'] = channel_wfm_fft_hp_lp

    return output_series

######################################################################################
def ParseTMSYwireWfm( raw_waveform, save_waveform=False ):

    output_series = pd.Series()

    # Denoising/filtering happens here.
    channel_fft = np.fft.rfft( raw_waveform )
    channel_fft[100:]=0
    channel_wfm_fft=np.fft.irfft(channel_fft)
    channel_wfm_fft_hp=digi_hp(channel_wfm_fft)
    channel_wfm_fft_hp_lp=digi_lp(channel_wfm_fft_hp)

    graph_fftcut=ROOT.TGraph()
    for isamp in xrange( len(raw_waveform) ):
        x = isamp
        #graph_fftcut.SetPoint(isamp,x,channel_wfm_lp[isamp])
        graph_fftcut.SetPoint(isamp,x,channel_wfm_fft_hp_lp[isamp])

    noise_wfm = np.array([graph_fftcut.GetY()[isamp] for isamp in xrange(0,200,1)])
    signal_wfm = np.array([graph_fftcut.GetY()[isamp] for isamp in xrange(200,900,1)])

    noise_baseline = np.mean(noise_wfm)
    noise_rms=np.sqrt(noise_wfm.dot(noise_wfm)/noise_wfm.size)
           
    signal_amp = np.min(signal_wfm)

    # fit_init is a tuple 
    fit_init_list = list()
    fit_init_list.append( 1. )             # integral
    fit_init_list.append( float(300.0) )  # starting time
    #fit_init_list.append( float(125.*1) )  # differential time
    #fit_init_list.append( float(31.25*2))  # integration time
    fit_init_list.append( noise_baseline)  # noise floor

    if (noise_baseline - signal_amp)>3*noise_rms and (np.argmin(signal_wfm)+300)>300 :

         #fitFunc.SetParameter(0,np.max(signal_wfm)*100)

         fit_init_list[0] = np.min(signal_wfm)*8.
         fit_init = tuple( fit_init_list )
         #print('Init: [{},{},{}]'.format(fit_init_list[0],\
         #                                fit_init_list[1],\
         #                                fit_init_list[2]))

         fit_start = 200#np.argmax(signal_wfm)+2000-150
         fit_end   = 900#np.argmax(signal_wfm)+2000+400

         fit_mask = range(fit_start,fit_end+1)
         
         fit_x = np.arange(float(fit_start),float(fit_end+1))
         fit_y = channel_wfm_fft_hp_lp[fit_mask]  
#         print('fit_x: {}'.format(fit_x))
#         print('fit_y: {}'.format(fit_y))

         try:
           popt, pcov = opt.curve_fit( fitShapedConst, \
                                       fit_x, \
                                       fit_y, \
                                       p0=fit_init_list )
         except RuntimeError as e:
           print(e)
           popt = np.zeros(len(fit_init))
           pcov = np.zeros(len(fit_init))
         #print('Fit: [{},{},{}]'.format(popt[0],popt[1],popt[2]))
#         print('Popt {}'.format(popt))
#         print('Pcov {}'.format(pcov))
#         print('pcov.shape {}'.format(pcov.shape))
#         if pcov.shape
         try:
           perr = np.sqrt( np.diag( pcov ) )
         except ValueError:
           perr = np.zeros(len(popt))


         y_from_fit = fitShapedConst( fit_x, popt[0], popt[1], popt[2] )
         redChi2 = np.sum( (y_from_fit - fit_y)**2 ) / ( len(fit_x) - len(popt) )

         deltaT=0.1*(popt[2]-np.min(y_from_fit))*60
         startT= popt[1]

         if(startT>300) and (startT-300)*0.052<20 and redChi2 <5 :
                output_series['Drift'] = (startT-300)*0.052
                output_series['Energy'] = signal_amp
                output_series['Drift Err'] = perr[1]*0.052
                output_series['Fit RedChi2'] = redChi2
         else: 
                output_series['Drift'] = -1.
                output_series['Energy'] = -1.
                output_series['Drift Err'] = -1.
                output_series['Fit RedChi2'] = redChi2

    else:
         output_series['Drift'] = -1.
         output_series['Drift Err'] = -1.
         output_series['Energy'] = -1.
         output_series['Fit RedChi2'] = -1.

    if save_waveform:
       output_series['TMS Data'] = raw_waveform
       output_series['TMS Denoised Data'] = channel_wfm_fft_hp_lp

    return output_series

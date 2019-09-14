import pandas as pd
import numpy as np
from scipy import optimize as opt
from tms_utilities.useful_function_shapes import GumbelDist
from tms_utilities.useful_function_shapes import DoubleExpGaussConv 
from tms_utilities.useful_function_shapes import DoubleExpGaussConvTFixed 


########################################################################################
def ParsePSDWfm( raw_waveform, save_waveform=False ):
 
  isNewEvent = True  

  reduced_data = pd.Series()

  if save_waveform:
     reduced_data['Data'] = raw_waveform

  reduced_data['Baseline'], reduced_data['Baseline RMS'] = GetBaseline( raw_waveform )    

  reduced_data.name = 'PSD Event'

  reduced_data['Pulse Height'],\
  reduced_data['Pulse Area'],\
  reduced_data['Fit Amplitude'],\
  reduced_data['Fit Frac'],\
  reduced_data['Fit Time'],\
  reduced_data['Fit Sig'],\
  reduced_data['Fit T1'],\
  reduced_data['Fit T2'],\
  reduced_data['PSD8'],\
  reduced_data['PSD9'],\
  reduced_data['PSD10'],\
  reduced_data['PSD11'] = FindPulseAndComputeArea( raw_waveform )

  return reduced_data
    
########################################################################################
def GetBaseline( data ):
  
   fullmean = np.mean( data[0:50] )
   fullrms = np.std( data[0:50] )
 
   mask = (data - fullmean)**2 < (5 * fullrms)**2

   baseline = np.mean( data[mask] )
   baselineRMS = np.std( data[mask] )
      
   return baseline, baselineRMS

########################################################################################
def FindPulseAndComputeArea( data ):

  baseline, baselineRMS = GetBaseline( data )
  
  threshold = 8*baselineRMS # 8-sigma threshold hardcoded in
  pre_nsamps = 10
  post_nsamps = 250
  
  pulse_idx = np.where( (data - baseline)**2 > (threshold)**2 )
  if len(pulse_idx[0]) == 0: return 0,0,0,0,0,0,0,0,0,0,0,0
  #print('Pulse array {}'.format(pulse_idx[0]) ) 
  #print('pulse_idx[0][-1] = {}, pulse_idx[0][0] = {}'.format(pulse_idx[0][-1],pulse_idx[0][0]))
  if (pulse_idx[0][-1] - pulse_idx[0][0])>(len(pulse_idx[0])-1 + pre_nsamps + post_nsamps):
     print('Multiple pulses found in event. Skipping...')
     return 0,0,0,0,0,0,0,0,0,0,0,0

  first_pulse = pulse_idx[0]
  start = first_pulse[0]-pre_nsamps
  end = first_pulse[-1]+post_nsamps

  if start < 0: start = 0
  if end > len(data): end = len(data)-1

  pulse_data = data[start:end] - baseline
  #print('Length of pulse_data: {}'.format(len(pulse_data)))

  #pulse_area = np.sqrt( np.sum( pulse_data )**2 )
  pulse_height = np.sqrt( np.max( pulse_data**2 ) )
  pulse_max_index = np.argmax( (data-baseline)**2 )
  pulse_area, aft_05 = GetPulseArea( pulse_data )
  x = np.linspace(start,end-1,end-start) - (aft_05 + start)
  i9 = np.where( x == 9. )
  i8 = np.where( x == 8. )
  i10 = np.where( x == 10. )
  i11 = np.where( x == 11. )
  psd8 = (np.cumsum(pulse_data)/pulse_area)[i8][0]
  psd9 = (np.cumsum(pulse_data)/pulse_area)[i9][0]
  psd10 = (np.cumsum(pulse_data)/pulse_area)[i10][0]
  psd11 = (np.cumsum(pulse_data)/pulse_area)[i11][0]

  # Initial guesses (based on high resolution scope traces)
  fit_A_0 = pulse_height*3.
  fit_B_0 = 0.8
  fit_mu_0 = pulse_max_index
  fit_sig_0 = 0.325
  fit_t1_0 = 0.75 # samples
  fit_t2_0 = 8. # samples

#  try: 
#    (fit_A, fit_B, fit_mu, fit_sig, fit_t1, fit_t2), _ = opt.curve_fit( DoubleExpGaussConv,\
#                                         np.linspace(start,end-1,(end-start)),\
#                                         pulse_data,\
#                                         p0=(fit_A_0,fit_B_0,fit_mu_0,fit_sig_0,fit_t1_0,fit_t2_0))
  try: 
    (fit_A, fit_B, fit_mu, fit_sig), _ = opt.curve_fit( DoubleExpGaussConvTFixed,\
                                         np.linspace(start,end-1,(end-start)),\
                                         pulse_data,\
                                         p0=(fit_A_0,fit_B_0,fit_mu_0,fit_sig_0))
    fit_t1 = 0.918
    fit_t2 = 14.2
  except:
    print('Fit error.')
    return 0,0,0,0,0,0,0,0,0,0,0,0


  return pulse_height, pulse_area, fit_A, fit_B, fit_mu, fit_sig, fit_t1, fit_t2, psd8, psd9, psd10, psd11

#######################################################################################
def GetPulseArea( data ):
    if len(data) == 0: return 0,0
#    peakidx = np.argmax(data)
#    if peakidx-10 < 0 or peakidx+250 > len(data): return 0,0
#    pulse = data[peakidx-10:peakidx+250]
    cumul_pulse = np.cumsum(data)
    pulse_area = cumul_pulse[-1]
    t0s = np.where( cumul_pulse > 0.05*pulse_area)[0][0]
    #print('t0s = {}'.format(t0s))
    #m = cumul_pulse[t0s]-cumul_pulse[t0s-1]
    #b = cumul_pulse[t0s] - m*t0s
    aft_05 = t0s #+ ( 0.05*pulse_area - b)/m
    #print(pulse_area)
    #print(aft_05)
    return pulse_area, aft_05




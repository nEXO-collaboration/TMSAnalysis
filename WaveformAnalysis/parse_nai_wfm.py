import pandas as pd
import numpy as np
from scipy import optimize as opt
from tms_utilities.useful_function_shapes import TwoExpConv
from tms_utilities.useful_function_shapes import Gaussian

########################################################################################
def ParseNaIWfm( raw_waveform, save_waveform=False ):
 
  reduced_data = pd.Series()

  if save_waveform:
     reduced_data['Data'] = raw_waveform

  reduced_data['Baseline'], reduced_data['Baseline RMS'] = GetBaseline( raw_waveform )    
  reduced_data.name = 'NaI Event'

  try:
    reduced_data['Pulse Height'],\
    reduced_data['Pulse Time']  = NaICrossCorrelationPulseFinder( raw_waveform - reduced_data['Baseline'] )
  except ValueError as e:
    print(e)
    reduced_data['Pulse Height'] = float('nan')
    reduced_data['Pulse Time'] = float('nan')

#  reduced_data['Data (BSS)'] = raw_waveform - reduced_data['Baseline']

  return reduced_data
    
########################################################################################
def GetBaseline( data ):

   # Baseline is computed from first 50 samples. This may need to be fine-tuned.  

   fullmean = np.mean(data[0:50])
   fullrms = np.std(data[0:50])
   mask = (data - fullmean)**2 < (6 * fullrms)**2
 
   baseline = np.mean(data[mask])
   baselineRMS = np.std(data[mask])

   #print('{} {} {} {}'.format(tempmean,tempRMS,baseline,baselineRMS))
      
   return baseline, baselineRMS


#######################################################################################
def DoubleExpConvConstNaI(x,A,mu):                                                                                        

    # These constants will need to be fine-tuned for the new digitizer

    t1 = 3.0
    t2 = 30.0
    mask = (x-mu)<0
    y = A/(t1-t2)*( np.exp(-(x-mu)/t1) - np.exp(-(x-mu)/t2) )
    y[mask] = np.zeros(len(y))[mask]
    return y

#######################################################################################
def NaICrossCorrelation( x, y, trigger_position, cc_window):
    cross_cor = np.zeros(len(y))
    

    for i in range(trigger_position-cc_window,trigger_position+cc_window):
        cross_cor[i] = np.sum( DoubleExpConvConstNaI(x,-1.,float(i))*y )

    return cross_cor*1.654311 # Empirical scaling factor so that the cross correlation 
                              # height equals the true pulse height in ADC counts.

#######################################################################################
def NaICrossCorrelationPulseFinder( data ):

    trigger_position = 2000 # sample number of trigger within waveform
    cc_window = 250 # number of samples on either side of trigger to include in cross-correlation
    cut_window = 100 # number of samples on either side of trigger to search for pulses

    x = np.linspace(0.,len(data)-1.,len(data))
    y = data               

    cross_cor = NaICrossCorrelation( x, y, trigger_position, cc_window)
    pulse_heights = np.array([])                
    pulse_times = np.array([])
    pulse_idxs = np.array([])
    
    pulse_idxs, pulse_heights = FindLocalMaxima( cross_cor, trigger_position, cc_window )
    print('Pulse idxs: {}'.format(pulse_idxs))
    cut = (pulse_idxs-trigger_position)**2 < cut_window**2
    #print('Cut: {}'.format(cut))
    pulse_heights = pulse_heights[ cut ]
    pulse_idxs = pulse_idxs[ cut ]
    print('Pulse idxs: {}'.format(pulse_idxs))
    
 
    if len(pulse_idxs) > 1:
          #print('Pulse indexes: {}'.format(pulse_idxs))
          print('Multiple pulses found in event. Skipping...')
          return  0,0
    if len(pulse_idxs) == 0:
        raise ValueError('No pulses found in event')
        return 0,0
        
    start = int(pulse_idxs[0]) - 2
    end = int(pulse_idxs[0]) + 4
    try:     
      (Afit, mufit, sigfit),_ = opt.curve_fit( Gaussian,\
                                              x[start:end],\
                                              cross_cor[start:end],\
                                              p0=(pulse_heights[0],pulse_idxs[0],11.) )
    except RuntimeError as e:
      print(e)
      Afit = 0.
      mufit = 0.
      sigfit = 0.

    return Afit, mufit
    
    
#######################################################################################
def FindLocalMaxima(data, trigger_position, cc_window):
    mask = data>13.
    local_maxima = np.array([])
    values = np.array([])
    for i in range(trigger_position - cc_window,trigger_position + cc_window):
        if data[i] > data[i-1] and data[i] > data[i+1] and\
           data[i] > data[i-2] and data[i] > data[i+2] and\
           data[i+1] > data[i+2] and data[i-1] > data[i-2] and \
           mask[i]:
            local_maxima = np.append(local_maxima,i)
            values = np.append(values,data[i])
    return local_maxima, values     
                   

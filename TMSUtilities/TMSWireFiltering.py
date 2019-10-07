import numpy as np
from numba import jit


@jit(nopython=True,parallel=True)
def DigiHighPass( waveform ):

    # waveform should be a numpy array containing a waveform
    # also it's a differentiator

    rcconst = 1 #micro seconds
    deltat = 0.008 #micro seconds
 
    alpha = rcconst/(rcconst+deltat)
    y = np.zeros(len(waveform))
  
    for i in range(1,len(waveform)):
        y[i] = alpha*(waveform[i] - waveform[i-1]) + alpha*y[i-1]
    return y

@jit(nopython=True,parallel=True)
def DigiLowPass( waveform ):

    # waveform should be a numpy array containing a waveform
    # also it's an integrator
 
    rcconst = 0.25 #micro seconds
    deltat = 0.008 #micro seconds
 
    alpha = deltat/(rcconst+deltat)

    y = np.zeros(len(waveform))
  
    for i in range(1,len(waveform)):
        y[i] = alpha*(waveform[i]) + (1-alpha)*y[i-1]
    return y


def WaveformFFT( waveform, sampling_period ):
    
    wfm_fft = np.fft.rfft(waveform)
    fft_filter_array = np.zeros_like( wfm_fft )   

    sampling_freq_Hz=1./(sampling_period*1.e-9)
    fft_freq = np.fft.rfftfreq(len(waveform),d=1./sampling_freq_Hz)                                                       
    ch_low_pass = 8.0e6                                                                                                       
    fft_freq_pass = np.logical_and(fft_freq > 5e3, fft_freq < ch_low_pass)

    fft_filter_array[fft_freq_pass] = wfm_fft[fft_freq_pass] 
    wfm_fft = np.fft.irfft(fft_filter_array)

    return wfm_fft


def WaveformFFTAndFilter( waveform, sampling_period ):

    
    wfm_fft = np.fft.rfft(waveform)
    fft_filter_array = np.zeros_like( wfm_fft )   

    sampling_freq_Hz=1./(sampling_period*1.e-9)
    fft_freq = np.fft.rfftfreq(len(waveform),d=1./sampling_freq_Hz)                                                       
    ch_low_pass = 8.0e6                                                                                                       
    fft_freq_pass = np.logical_and(fft_freq > 5e3, fft_freq < ch_low_pass)

    fft_filter_array[fft_freq_pass] = wfm_fft[fft_freq_pass] 
    wfm_fft = np.fft.irfft(fft_filter_array)   
    wfm_fft_hp = DigiHighPass(wfm_fft)
    wfm_fft_hp_lp = DigiLowPass(wfm_fft_hp)

    return wfm_fft_hp_lp

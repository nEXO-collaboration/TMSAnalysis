import numpy as np

##################################################################
def digi_hp( waveform ):

  # waveform should be a numpy array containing a waveform

  rcconst = 2 #micro seconds
  deltat = 0.016 #micro seconds
 
  alpha = rcconst/(rcconst+deltat)

  y = np.zeros(len(waveform))
  
  #y[0]=waveform[0]

  for i in range(1,len(waveform)):
      y[i] = alpha*(waveform[i] - waveform[i-1]) + alpha*y[i-1]
    
  return y

##################################################################

def digi_lp( waveform ):

  # waveform should be a numpy array containing a waveform

  rcconst = 1 #micro seconds
  deltat = 0.016 #micro seconds
 
  alpha = deltat/(rcconst+deltat)

  y = np.zeros(len(waveform))
  
  #y[0]=waveform[0]

  for i in range(1,len(waveform)):
      y[i] = alpha*(waveform[i]) + (1-alpha)*y[i-1]
    
  return y

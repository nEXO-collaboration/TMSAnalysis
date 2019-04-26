########################################################################
# This file defines a Waveform class that allows us to do general
# waveform-level analysis. Specific routines for analyzing different
# detectors will be written in separate files.
#   
#    - Brian L.
########################################################################

import pandas as pd
import numpy as np


class Waveform:

	def __init__( self, input_data=None, detector_type=None, sampling_rate=None ):
		self.data = input_data
		self.detetor_type = detector_type
		self.sampling_rate = sampling_rate
		self.analysis_quantities = pd.Series()

	def DataCheck( self ):
		if self.data is not None:
			return
		else:
			raise Exception('No data in waveform.')

	def FindPulsesAndComputeArea( self ):
		self.DataCheck()
		baseline = np.mean(self.data[0:50])
		baseline_rms = np.std(self.data[0:50])		
		self.analysis_quantities['Baseline'] = baseline
		self.analysis_quantities['Baseline RMS'] = baseline_rms
		self.analysis_quantities['Num Pulses'] = 0
		self.analysis_quantities['Pulse Areas'] = np.array([])
		self.analysis_quantities['Pulse Times'] = np.array([])

		threshold = 10*baseline_rms
		pre_nsamps = 10
		post_nsamps = 70
		
		pulse_idx = np.where( (self.data-baseline)**2 > threshold**2 )
		# First, check if there are no pulses
		if len(pulse_idx[0]) == 0:
			return
		# Next, check if there are multiple pulses
		elif (pulse_idx[0][-1] - pulse_idx[0][0]) > \
		   (len(pulse_idx[0])-1 + pre_nsamps + post_nsamps):
			print('Multiple pulses found. This is not yet supported.')
			return
		# Finally, find the interesting characteristics of the pulse
		else:
			start = pulse_idx[0][0]-pre_nsamps
			end = pulse_idx[0][-1]+post_nsamps
			pulse_area, pulse_time = self.GetPulseArea( self.data[start:end]-baseline )
			self.analysis_quantities['Num Pulses'] = 1
			self.analysis_quantities['Pulse Areas'] = \
				np.append( self.analysis_quantities['Pulse Areas'], pulse_area )
			self.analysis_quantities['Pulse Times'] = \
				np.append( self.analysis_quantities['Pulse Times'], pulse_time+start )
	
	def GetPulseArea( self, dat_array ):
		if len(dat_array) == 0: return 0,0
		cumul_pulse = np.cumsum(dat_array)
		pulse_area = cumul_pulse[-1]
		if pulse_area < 0.:
			pulse_area = pulse_area*(-1.)
			cumul_pulse = cumul_pulse*(-1.)
		t0_5percent = np.where( cumul_pulse > 0.05*pulse_area)[0][0]
		return pulse_area, t0_5percent

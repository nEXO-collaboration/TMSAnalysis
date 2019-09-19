########################################################################
# This file defines a Waveform class that allows us to do general
# waveform-level analysis. Specific routines for analyzing different
# detectors will be written in separate files.
#   
#    - Brian L.
#
# Note: the waveform processings stuff will need to change based on the
# type of detector that we're recording data from; i.e. an NaI signal
# will need a different processing framework than a simple PMT signal
# looking for cherenkov light. 
########################################################################

import pandas as pd
import numpy as np



class Waveform:

	def __init__( self, input_data=None, detector_type=None, sampling_rate=None, \
			input_baseline=-1, input_baseline_rms=-1, \
			fixed_window=False, window_start=0, window_end=0 ):
		self.data = input_data
		self.input_baseline = input_baseline
		self.input_baseline_rms = input_baseline_rms
		self.detetor_type = detector_type
		self.fixed_window = fixed_window
		self.window_start = window_start
		self.window_end = window_end
		# Make the default detector type a simple PMT
		if detector_type == None:
			self.detector_type = 'PMT'
		if fixed_window:
			if window_start==window_end:
				print('***ERROR***: You\'ve selected a fixed_window analyis, but the window has zero length.')
		self.sampling_rate = sampling_rate
		self.analysis_quantities = pd.Series()

	def DataCheck( self ):
		if self.data is not None:
			return
		else:
			raise Exception('No data in waveform.')

	def FindPulsesAndComputeArea( self ):
		self.DataCheck()
		if self.input_baseline < 0.:
			baseline = np.mean(self.data[0:50])
			baseline_rms = np.std(self.data[0:50])
		else:
			baseline = self.input_baseline
			baseline_rms = self.input_baseline_rms		
		self.analysis_quantities['Baseline'] = baseline
		self.analysis_quantities['Baseline RMS'] = baseline_rms
		self.analysis_quantities['Num Pulses'] = 0
		self.analysis_quantities['Pulse Areas'] = np.array([])
		self.analysis_quantities['Pulse Heights'] = np.array([])
		self.analysis_quantities['Pulse Times'] = np.array([])

		if not self.fixed_window:
			threshold = 10*baseline_rms
			pre_nsamps = 10
			post_nsamps = 10
			if self.detector_type == 'NaI':
				pre_nsamps = 20
				post_nsamps = 100
			if self.detector_type == 'PMT':
				pre_nsamps = 5
				post_nsamps = 7
			
			pulse_idx = np.where( (self.data-baseline)**2 > threshold**2 )
			# First, check if there are no pulses
			if len(pulse_idx[0]) == 0:
				return
			# Next, check if there are multiple pulses
			elif (pulse_idx[0][-1] - pulse_idx[0][0]) > \
			   (len(pulse_idx[0])-1 + pre_nsamps + post_nsamps):
				print('Multiple pulses found in {} detector. This is not yet supported.'.format(self.detector_type))
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
				self.analysis_quantities['Pulse Heights'] = \
					np.append( self.analysis_quantities['Pulse Heights'], np.min(self.data[start:end]-baseline) )
		else:
			baseline = np.mean(self.data[self.window_start:self.window_start+10])
			pulse_area, pulse_time = self.GetPulseArea( self.data[self.window_start:self.window_end]-baseline )
			self.analysis_quantities['Num Pulses'] = 1
			self.analysis_quantities['Pulse Areas'] = \
				np.append( self.analysis_quantities['Pulse Areas'], pulse_area )
			self.analysis_quantities['Pulse Times'] = \
				np.append( self.analysis_quantities['Pulse Times'], pulse_time+self.window_start )
			self.analysis_quantities['Pulse Heights'] = \
				np.append( self.analysis_quantities['Pulse Heights'], np.min(self.data[self.window_start:self.window_end]-baseline) )
				


	def GetPulseArea( self, dat_array ):
		if len(dat_array) == 0: return 0,0
		cumul_pulse = np.cumsum(dat_array)
		pulse_area = np.mean(cumul_pulse[-4:-1])
		if pulse_area < 0.:
			pulse_area = pulse_area*(-1.)
			cumul_pulse = cumul_pulse*(-1.)
		t0_5percent = np.where( cumul_pulse > 0.05*pulse_area)[0][0]
		return pulse_area, t0_5percent

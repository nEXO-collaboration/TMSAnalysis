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
from TMSAnalysis.tms_utilities import useful_function_shapes as ufun
import scipy.optimize as opt

class Waveform:

	def __init__( self, input_data=None, detector_type=None, sampling_period=None, \
			input_baseline=-1, input_baseline_rms=-1, \
			fixed_window=False, window_start=0, window_end=0 ):
		self.data = input_data
		self.input_baseline = input_baseline
		self.input_baseline_rms = input_baseline_rms
		self.detector_type = detector_type
		self.fixed_window = fixed_window
		self.window_start = window_start
		self.window_end = window_end
		# Make the default detector type a simple PMT
		if detector_type == None:
			self.detector_type = 'PMT'
		if fixed_window:
			if window_start==window_end:
				print('***ERROR***: You\'ve selected a fixed_window analyis, but the window has zero length.')
		self.sampling_period = sampling_period
		self.analysis_quantities = pd.Series()

	def NaIPulseTemplate( self, x, amp, time):
		return ufun.TwoExpConv(x, amp*2.5, time+40./self.sampling_period, 58./self.sampling_period, 200.5/self.sampling_period)

	def PSPulseTemplate( self, x, amp, time):
		return ufun.DoubleExpGaussConv( x, amp*10.5, 0.76, time, \
						2./self.sampling_period, \
						6.5/self.sampling_period, \
						37.3/self.sampling_period )

	def CherenkovPulseTemplate( self, x, amp, time ):
		return ufun.DoubleExpGaussConv( x, amp * 6.7, 0.88, time, \
						1.8/self.sampling_period, \
						4.1/self.sampling_period, \
						49./self.sampling_period )  

	def DataCheck( self ):
		if self.data is not None:
			return
		else:
			raise Exception('No data in waveform.')

	def FindPulsesAndComputeArea( self, fit_pulse_flag=False ):
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
		self.analysis_quantities['Fit Heights'] = np.array([])
		self.analysis_quantities['Fit Times'] = np.array([])

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
			fit_height = 0.
			fit_time = 0.
			if 'NaI' in self.detector_type:
				baseline = np.mean(self.data[self.window_start-25:self.window_start-15])
				baseline_rms = np.std(self.data[self.window_start-25:self.window_start+10])
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( self.data[self.window_start-25:self.window_end+90]-baseline )
				if fit_pulse_flag == True:
					xwfm = np.linspace(0.,(self.window_end-self.window_start)+115-1,(self.window_end-self.window_start)+115)
					popt,pcov = opt.curve_fit( self.NaIPulseTemplate, xwfm, self.data[self.window_start-25:self.window_end+90]-baseline,\
									p0=(pulse_height*7.,pulse_time),xtol=0.05,ftol=0.05)
					fit_height = popt[0]
					fit_time = popt[1]
				pulse_time = pulse_time - 25
				fit_time = fit_time-25
			elif 'Cherenkov' in self.detector_type:
				baseline = np.mean(self.data[self.window_start+2:self.window_start+12])
				baseline_rms = np.std(self.data[self.window_start+2:self.window_start+12])
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( self.data[self.window_start+2:self.window_end-5]-baseline )
				if (fit_pulse_flag == True) and (pulse_height < 7.):
					xwfm = np.linspace(0.,(self.window_end-self.window_start)-1,(self.window_end-self.window_start))
					popt,pcov = opt.curve_fit( self.CherenkovPulseTemplate, xwfm, self.data[self.window_start:self.window_end]-baseline,\
									p0=(pulse_height,pulse_time),xtol=0.001,ftol=0.001)
					fit_height = popt[0]
					fit_time = popt[1]
			elif 'PS' in self.detector_type:
				baseline = np.mean(self.data[self.window_start:self.window_start+10])
				baseline_rms = np.std(self.data[self.window_start:self.window_start+10])
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( self.data[self.window_start:self.window_end]-baseline )
				if fit_pulse_flag == True:
					xwfm = np.linspace(0.,(self.window_end-self.window_start)-1,(self.window_end-self.window_start))
					popt,pcov = opt.curve_fit( self.PSPulseTemplate, xwfm, self.data[self.window_start:self.window_end]-baseline,\
									p0=(pulse_height,pulse_time),xtol=0.002,ftol=0.002)
					fit_height = popt[0]
					fit_time = popt[1]
			else:
				pulse_area = 0.
				pulse_time = 0.
				pulse_height = 0.
			self.analysis_quantities['Baseline'] = baseline
			self.analysis_quantities['Baseline RMS'] = baseline_rms
			self.analysis_quantities['Num Pulses'] = 1
			self.analysis_quantities['Pulse Areas'] = \
				np.append( self.analysis_quantities['Pulse Areas'], pulse_area )
			self.analysis_quantities['Pulse Times'] = \
				np.append( self.analysis_quantities['Pulse Times'], pulse_time+self.window_start )
			self.analysis_quantities['Pulse Heights'] = \
				np.append( self.analysis_quantities['Pulse Heights'], pulse_height )
			self.analysis_quantities['Fit Heights'] = \
				np.append( self.analysis_quantities['Fit Heights'], fit_height )
			self.analysis_quantities['Fit Times'] = \
				np.append( self.analysis_quantities['Fit Times'], fit_time )
				


	def GetPulseArea( self, dat_array ):
		if len(dat_array) == 0: return 0,0
		cumul_pulse = np.cumsum(dat_array)
		pulse_area = np.mean(cumul_pulse[-4:-1])
		if pulse_area < 0.:
			pulse_area = pulse_area*(-1.)
			cumul_pulse = cumul_pulse*(-1.)
		t0_10percent_samp = np.where( cumul_pulse > 0.1*pulse_area)[0][0]
		# The next chunk does a linear interpolation to get the pulse time more accurately.
		t0_10percent = ( 0.1*pulse_area - cumul_pulse[t0_10percent_samp] + \
				t0_10percent_samp * \
				(cumul_pulse[t0_10percent_samp]-cumul_pulse[t0_10percent_samp-1]) ) /\
				(cumul_pulse[t0_10percent_samp]-cumul_pulse[t0_10percent_samp-1])
		pulse_height = np.min( dat_array )

		return pulse_area, t0_10percent, pulse_height

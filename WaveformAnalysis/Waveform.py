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
from TMSAnalysis.TMSUtilities import UsefulFunctionShapes as Ufun
from TMSAnalysis.TMSUtilities import TMSWireFiltering as Filter
import scipy.optimize as opt

class Waveform:

	def __init__( self, input_data=None, detector_type=None, sampling_period=None, \
			input_baseline=-1, input_baseline_rms=-1, polarity=-1., \
			fixed_trigger=False, trigger_position=0 ):
		self.data = input_data
		self.input_baseline = input_baseline
		self.input_baseline_rms = input_baseline_rms
		self.detector_type = detector_type
		self.fixed_trigger = fixed_trigger
		self.trigger_position = trigger_position
		self.polarity = polarity
		# Make the default detector type a simple PMT
		if detector_type == None:
			self.detector_type = 'PMT'
		#if fixed_window:
		#	if window_start==window_end:
		#		print('***ERROR***: You\'ve selected a fixed_window analyis, but the window has zero length.')
		self.sampling_period = sampling_period
		self.analysis_quantities = dict()

	def NaIPulseTemplate( self, x, amp, time):
		return Ufun.TwoExpConv(x, amp*30., time-40./self.sampling_period, 58./self.sampling_period, 200.5/self.sampling_period)

	def PSPulseTemplate( self, x, amp, time):
		return Ufun.DoubleExpGaussConv( x, amp*2., 0.80, time + 5./self.sampling_period, \
						2./self.sampling_period, \
						6.5/self.sampling_period, \
						37.3/self.sampling_period )

	def CherenkovPulseTemplate( self, x, amp, time ):
		return Ufun.DoubleExpGaussConv( x, amp * 6.7, 0.90, time, \
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

		if not self.fixed_trigger:
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
				window_start = self.trigger_position - int(800/self.sampling_period)
				window_end = self.trigger_position + int(1600/self.sampling_period)
				baseline = np.mean(self.data[window_start:window_start+10])
				baseline_rms = np.std(self.data[window_start:window_start+10])
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( self.data[window_start:window_end]-baseline )
				if fit_pulse_flag == True and np.abs(pulse_height)>10.*baseline_rms:
					xwfm = np.linspace(0.,(window_end-window_start)-1,(window_end-window_start))
					popt,pcov = opt.curve_fit( self.NaIPulseTemplate, xwfm, self.data[window_start:window_end]-baseline,\
									p0=(pulse_height*7.,pulse_time),xtol=0.05,ftol=0.05)
					fit_height = popt[0]
					fit_time = popt[1]
				pulse_time = pulse_time - int(800/self.sampling_period)
				fit_time = fit_time - int(800/self.sampling_period)
			elif 'Cherenkov' in self.detector_type:
				window_start = self.trigger_position - int(320/self.sampling_period)
				window_end = self.trigger_position + int(160/self.sampling_period)
				baseline = np.mean(self.data[window_start:window_start+10])
				baseline_rms = np.std(self.data[window_start:window_start+10])
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( self.data[window_start:window_end]-baseline )
				if (fit_pulse_flag == True) and (pulse_height < 7.) and np.abs(pulse_height)>10.*baseline_rms:
					xwfm = np.linspace(0.,(window_end-window_start)-1,(window_end-window_start))
					popt,pcov = opt.curve_fit( self.CherenkovPulseTemplate, xwfm, self.data[window_start:window_end]-baseline,\
									p0=(pulse_height,pulse_time),xtol=0.001,ftol=0.001)
					fit_height = popt[0]
					fit_time = popt[1]
				pulse_time = pulse_time - int(320/self.sampling_period)
				fit_time = fit_time - int(320/self.sampling_period)
			elif 'PS' in self.detector_type:
				window_start = self.trigger_position - int(400/self.sampling_period)
				window_end = self.trigger_position + int(160/self.sampling_period)
				baseline = np.mean(self.data[window_start:window_start+10])
				baseline_rms = np.std(self.data[window_start:window_start+10])
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( self.data[window_start:window_end]-baseline )
				if fit_pulse_flag == True and np.abs(pulse_height>10.*baseline_rms):
					xwfm = np.linspace(0.,(window_end-window_start)-1,(window_end-window_start))
					popt,pcov = opt.curve_fit( self.PSPulseTemplate, xwfm, self.data[window_start:window_end]-baseline,\
									p0=(pulse_height,pulse_time),xtol=0.002,ftol=0.002)
					fit_height = popt[0]
					fit_time = popt[1]
				pulse_time = pulse_time - int(400/self.sampling_period)
				fit_time = fit_time - int(400/self.sampling_period)
			elif 'Xwire' in self.detector_type:
				self.polarity = (-1.)*self.polarity		
				window_start = self.trigger_position - int(2400/self.sampling_period) # 2.4us pretrigger
				window_end = self.trigger_position + int(10000/self.sampling_period)  # 10us posttrigger

				baseline = np.mean(self.data[0:250])
				fft_wfm = Filter.WaveformFFT( self.data-baseline, 8. )
				filtered_wfm = Filter.WaveformFFTAndFilter( self.data - baseline , 8. )	
				self.analysis_quantities['RawEnergy'] = np.mean( fft_wfm[window_end:window_end+300] ) - \
									np.mean( fft_wfm[window_start-200:window_start] )
				baseline = np.mean(filtered_wfm[-500:-1])
				baseline_rms = np.std(filtered_wfm[-500:-1])
				# Pulse time, area, height, position are derived from the filtered waveform.
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( filtered_wfm[window_start:window_end] )
				pulse_time = pulse_time - int(2400/self.sampling_period)
			elif 'Ywire' in self.detector_type:
				self.polarity = (-1.)*self.polarity		
				window_start = self.trigger_position - int(2400/self.sampling_period) # 2.4us pretrigger
				window_end = self.trigger_position + int(10000/self.sampling_period)  # 10us posttrigger

				baseline = np.mean(self.data[0:250])
				fft_wfm = Filter.WaveformFFT( self.data-baseline, 8. )
				filtered_wfm = Filter.WaveformFFTAndFilter( self.data - baseline , 8. )	
				self.analysis_quantities['RawEnergy'] = np.mean( fft_wfm[window_end:window_end+300] ) - \
									np.mean( fft_wfm[window_start-200:window_start] )
				baseline = np.mean(filtered_wfm[-500:-1])
				baseline_rms = np.std(filtered_wfm[-500:-1])
				# Pulse time, area, height, position are derived from the filtered waveform.
				pulse_area, pulse_time, pulse_height = self.GetPulseArea( filtered_wfm[window_start:window_end] )
				pulse_time = pulse_time - int(2400/self.sampling_period)
			elif 'SiPM' in self.detector_type:
                                window_start = self.trigger_position - int(800/self.sampling_period)
                                window_end = self.trigger_position + int(1600/self.sampling_period)
                                baseline = np.mean(self.data[window_start:window_start+10])
                                baseline_rms = np.std(self.data[window_start:window_start+10])
                                pulse_area, pulse_time, pulse_height = self.GetPulseArea( self.data[window_start:window_end]-baseline )
                                pulse_time = pulse_time - int(800/self.sampling_period)

			elif 'TileStrip' in self.detector_type:
				pulse_area = 0.
				pulse_time = 0.
				pulse_height = 0.
				maw_length = 125 # Moving average over XX samples, corresponding to XX*0.008 us
				baseline = np.mean(self.data[0:950])
				#smooth_wfm = np.convolve(self.data-baseline,np.ones(maw_length))[0:-(maw_length-1)]/maw_length
				smooth_wfm = self.RunningMean( self.data-baseline, maw_length )
				baseline_rms = np.std(smooth_wfm[0:950])
				if np.mean(smooth_wfm[12000:])**2 > (5.*baseline_rms)**2:
					samps_above_threshold = np.where( smooth_wfm**2 > (5.*baseline_rms)**2 )[0]
					diff_above_threshold = samps_above_threshold[1:] - samps_above_threshold[0:-1]
					if np.sum( diff_above_threshold > 125. ) > 0:   # Effectively cuts out waveforms with large baseline fluctuations
						pulse_area = 0.
						pulse_time = 0.
						fit_height = 0.
						pulse_height = 0.
					else:
						pulse_area = np.mean(smooth_wfm[12000:])
						pulse_time = np.where(smooth_wfm < 0.9*np.min(smooth_wfm))[0][0]
						pulse_height = 0.
						fit_height = 0.
				if '2' in self.detector_type:
					pulse_area = pulse_area * 2.
			else:								
				pulse_area = 0.
				pulse_time = 0.
				pulse_height = 0.
			self.analysis_quantities['Baseline'] = baseline
			self.analysis_quantities['Baseline RMS'] = baseline_rms
			self.analysis_quantities['Num Pulses'] = 1
			self.analysis_quantities['Pulse Areas'] = pulse_area
			#self.analysis_quantities['Pulse Areas'] = \
				#np.append( self.analysis_quantities['Pulse Areas'], pulse_area )
			self.analysis_quantities['Pulse Times'] = pulse_time + self.trigger_position
			#self.analysis_quantities['Pulse Times'] = \
				#np.append( self.analysis_quantities['Pulse Times'], pulse_time + self.trigger_position )
			self.analysis_quantities['Pulse Heights'] = pulse_height 
			#self.analysis_quantities['Pulse Heights'] = \
				#np.append( self.analysis_quantities['Pulse Heights'], pulse_height )
			self.analysis_quantities['Fit Heights'] = fit_height
			#self.analysis_quantities['Fit Heights'] = \
				#np.append( self.analysis_quantities['Fit Heights'], fit_height )
			self.analysis_quantities['Fit Times'] = fit_time + self.trigger_position
			#self.analysis_quantities['Fit Times'] = \
				#np.append( self.analysis_quantities['Fit Times'], fit_time + self.trigger_position )
				


	def GetPulseArea( self, dat_array ):
		if len(dat_array) == 0: return 0,0
		cumul_pulse = np.cumsum( dat_array * self.polarity )
		pulse_area = np.mean(cumul_pulse[-4:-1])
		try:
			t0_10percent_samp = np.where( cumul_pulse > 0.1*pulse_area)[0][0]
		except IndexError:
			t0_10percent_samp = 1
		# The next chunk does a linear interpolation to get the pulse time more accurately.
		t0_10percent = ( 0.1*pulse_area - cumul_pulse[t0_10percent_samp] + \
				t0_10percent_samp * \
				(cumul_pulse[t0_10percent_samp]-cumul_pulse[t0_10percent_samp-1]) ) /\
				(cumul_pulse[t0_10percent_samp]-cumul_pulse[t0_10percent_samp-1])
		pulse_height = self.polarity * np.max( np.abs(dat_array) )

		return pulse_area, t0_10percent, pulse_height

	def RunningMean( self, dat_array, window_length ):
		cumsum = np.cumsum(np.insert(dat_array, 0, 0)) 
		return (cumsum[window_length:] - cumsum[:-window_length]) / float(window_length)

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
#
#
# 12 March 2020
# class Event added. Given an event number this class bundles the
# information from the reduced h5 file and the tier1 root file.
# For each channel the event has a waveform object. This class
# is meant to plot/smooth the waveforms of a specific event.
# It is possible to pass directly a tier1 file and event, without
# the reduced file. In this case the high level information
# usually extracted from the reduced file (energy and risetime)
# won't be computed niether plotted.
# Additional functionalities can be added
#
#   Jacopo
#
########################################################################

from StanfordTPCAnalysis.TMSUtilities import UsefulFunctionShapes as Ufun
from StanfordTPCAnalysis.TMSUtilities import TMSWireFiltering as Filter
from scipy.ndimage import gaussian_filter
import scipy.optimize as opt
from numba import jit
import pandas as pd
import numpy as np
import sys
import copy



class Waveform:

	#######################################################################################
	def __init__( self, analysis_config, input_data=None, sw_ch=None, detector_type=None):

		self.data = input_data				 # <- Waveform in numpy array
		self.is_simulation = analysis_config.run_parameters['Is Simulation']

		#we're going to be accessing analysis_config.run_parameters quite a bit in the
		#functions below; lets give it a pnemonic here for quick access. 
		self.conf = analysis_config.run_parameters
		self.analysis_config = analysis_config
		self.sw_ch = sw_ch

		#aparently the struck auto inverts with clock = 125 MHz. 
		if(analysis_config.run_parameters['DoInvert']):
			self.polarity = -1
		else:
			self.polarity = 1
		self.flag = False

		# Make the default detector type a simple PMT
		if detector_type == None:
			self.detector_type = 'PMT'
		else:
			self.detector_type = detector_type

		# All returned quantities will be stored in this
		# dict and added to the output dataframe in DataReduction.py
		self.analysis_quantities = {}


	#######################################################################################
	def FindPulsesAndComputeAQs(self):
		if(self.data is None):
			raise Exception('No data in waveform.')
			
		# NOTE: almost all analyses are fixed_trigger analyses, so the if statement
		#	should generally be true.

		if self.conf['Fixed Trigger']:

			# Here we have different processing algorithms for different sensor channels.
			if 'SiPM' in self.detector_type:
				try:
					self.data = self.data.to_numpy().astype(float) * self.polarity
				except AttributeError:
					self.data = self.data.astype(float) * self.polarity
				baseline_start_idx = self.conf['SiPM Baseline Start [samples]']
				baseline_end_idx = self.conf['SiPM Baseline Length [samples]']
				trigger_idx = self.conf['SiPM Pretrigger Length [samples]']

				#check to make sure that all of these sample indices are in the data
				if(len(self.data) < trigger_idx or len(self.data) < baseline_end_idx \
					or len(self.data) < baseline_start_idx or baseline_start_idx > baseline_end_idx):
					print("ERROR: Check your SiPM baseline parameters in Run_Parameters")
					sys.exit()

				baseline_data = self.data[baseline_start_idx:baseline_end_idx]
				self.baseline = np.mean(baseline_data)
				self.baseline_rms = np.std(baseline_data) #only is RMS if perfect gaussian noise
				self.corrected_data = (self.data - self.baseline)
				self.analysis_quantities['Baseline'] = self.baseline
				self.analysis_quantities['Baseline RMS'] = self.baseline_rms
				self.analysis_quantities['Peak Time'] = np.argmax(self.corrected_data)
				pulse_area, pulse_height, t5, t10, t25, t50, t90 = \
					self.GetPulseAreaAndTimingParameters( self.corrected_data )
				self.analysis_quantities['Pulse Area'] = pulse_area
				self.analysis_quantities['Pulse Height'] = pulse_height
				self.analysis_quantities['T5'] = t5
				self.analysis_quantities['T10'] = t10
				self.analysis_quantities['T25'] = t25
				self.analysis_quantities['T50'] = t50
				self.analysis_quantities['T90'] = t90

				if(pulse_height > self.conf['SiPM Threshold [sigma]']*self.baseline_rms):
					self.analysis_quantities['Passes Threshold'] = True
				else:
					self.analysis_quantities['Passes Threshold'] = False

			elif 'TileStrip' in self.detector_type:
				######################### 2022/08/16 Jacopo testing the old charge recon algorithm on Run34/DS08
				## 'TileStrip' denotes the strips read out with the
				## charge-sensitive discrete preamps.

				##this is the smoothing time window in ns
				#ns_smoothing_window = 500.0

				## raw, unfiltered, uncorrected data, with polarity flipped if necessary
				#self.data = self.data.to_numpy().astype(float) * self.polarity

				
				# smooth data with Gaussian filter
				#self.data = gaussian_filter( self.data, \
				#			     ns_smoothing_window/self.sampling_period_ns )

				# get channel baseline after smoothing but before decay time correction or filtering
				# mostly used for shifting waveforms when plotting, so NOT corrected using
				# channel calibration constant
				#baseline = np.mean(self.data[0:self.input_baseline])

				# apply decay time correction to smoothed data
				#self.corrected_data = DecayTimeCorrection( self.data - baseline, \
				#					   self.decay_time_us, \
				#					   self.sampling_period_ns )

				## apply differentiator to filter out low-frequency noise from corrected data
				#differentiator_window = 100
				#self.corrected_data = Differentiator( self.corrected_data, differentiator_window)

				## get baseline after filtering to subtract off before energy is calculated
				#corrected_baseline = np.mean(self.corrected_data[0:self.input_baseline])

				## baseline RMS is calculated from filtered waveform so that charge energy and
				## baseline RMS can be compared directly to one another
				#baseline_rms = np.std(self.corrected_data[0:self.input_baseline])

				## set window lengths
				#induction_window_ns = 4000
				#ind_window_sample = int(induction_window_ns/self.sampling_period_ns)
				#window_length_us = 5. # calculate energy from last 5 us of cumulative pulse area

				## scale baseline RMS for integrated filtered waveform
				#area_window_sample = int(window_length_us*1000./self.sampling_period_ns)
				## factor of 1.5 corrects for non-gaussianity of noise on integrated waveform
				##area_baseline_rms = 1.5*baseline_rms*np.sqrt(area_window_sample)
				
				## now calculate pulse area, pulse height, and timing parameters from filtered waveform
				#pulse_area, pulse_height, t5, t10, t25, t50, t90 = \
				#	self.GetPulseAreaAndTimingParameters( self.corrected_data - corrected_baseline, \
				#					      window_length_us )			    

				## can use pulse height or pulse area to compute charge energy
				#charge_energy = pulse_height
				
				#if (pulse_height > self.strip_threshold*baseline_rms) \
				#   & (pulse_area > 3*area_baseline_rms): # Compute timing/position if charge energy is positive and above noise.
				#	# Compute drift time in microseconds (sampling is given in ns)
				#	drift_time = (t90 - self.trigger_position) * (self.sampling_period_ns / 1.e3)
				#else:
				#	t5 = -1.
				#	t10 = -1.
				#	t25 = -1.
				#	t50 = -1.
				#	t90 = -1.
				#	drift_time = -1.

				## finally, apply calibration constant correction
				## note: calibration constants are defined to be the numbers that scale the
				## final energy on a channel, calculated from the smoothed, decay corrected,
				## differentiated, baseline-subtracted waveform, to the true energy deposited.
				## for this reason the calibration constants are applied to the baseline_rms,
				## pulse_height, pulse_area, and charge_energy at the end
				#self.corrected_data *= self.calibration_constant
				#charge_energy *= self.calibration_constant
				#pulse_height *= self.calibration_constant
				#pulse_area *= self.calibration_constant
				#baseline_rms *= self.calibration_constant
				#area_baseline_rms *= self.calibration_constant

				## save all AQs
				#self.analysis_quantities['Baseline'] = baseline
				#self.analysis_quantities['Baseline RMS'] = baseline_rms
				#self.analysis_quantities['Integrated RMS'] = area_baseline_rms
				#self.analysis_quantities['Charge Energy'] = charge_energy
				#self.analysis_quantities['Pulse Height'] = pulse_height
				#self.analysis_quantities['Pulse Area'] = pulse_area
				#self.analysis_quantities['T5'] = t5
				#self.analysis_quantities['T10'] = t10
				#self.analysis_quantities['T25'] = t25
				#self.analysis_quantities['T50'] = t50
				#self.analysis_quantities['T90'] = t90
				#self.analysis_quantities['Drift Time'] = drift_time
				#self.analysis_quantities['Induced Charge'] = self.Induction_Charge(ind_window_sample)
				########################################## OLD RECON STARTS HERE

				ns_smoothing_window = 500.0
				try:
					self.data = self.data.to_numpy().astype(float) * self.polarity
				except AttributeError:
					self.data = self.data.astype(float) * self.polarity

				self.data = gaussian_filter( self.data, ns_smoothing_window/self.sampling_period_ns )

				baseline_start_idx = self.conf['Charge Baseline Start [samples]']
				baseline_end_idx = self.conf['Charge Baseline Length [samples]']
				trigger_idx = self.conf['Charge Pretrigger Length [samples]']

				#check to make sure that all of these sample indices are in the data
				if(len(self.data) < trigger_idx or len(self.data) < baseline_end_idx \
					or len(self.data) < baseline_start_idx or baseline_start_idx > baseline_end_idx):
					print("ERROR: Check your Charge baseline parameters in Run_Parameters")
					sys.exit()

				baseline_data = self.data[baseline_start_idx:baseline_end_idx]
				self.baseline = np.mean(baseline_data)
				self.baseline_rms = np.std(baseline_data) #only is RMS if perfect gaussian noise

				#this adjusts the charge waveform to perform an exponential filter
				#that corrects for the decay time of the pre-amplifiers. 
				
				#get decay time from analysis config
				decay_time_us = self.analysis_config.GetDecayTimeForSoftwareChannel(self.sw_ch)
				self.corrected_data = DecayTimeCorrection(self.data - self.baseline, \
									decay_time_us, \
									self.conf['Sampling Period [ns]'])

				#at this point, charge energy is calculated using the last 
				#few microseconds of the exponential decay corrected waveform. 
				charge_energy_samples = int(self.conf['Charge Energy Calculation Time [us]']*1000/self.conf['Sampling Period [ns]'])      
				charge_energy = np.mean(self.corrected_data[-1*charge_energy_samples:])

				# ^Charge energy calculated from the last 5us of the smoothed, corrected wfm
				t10 = None
				t25 = None
				t50 = None
				t90 = None
				drift_time = None
				# Compute timing/position if charge energy is positive and above noise.
				# One adjustment has been made by Evan on 1/27/23, that I would like
				# to find the threshold crossing that happens closest to the maximum
				# of the charge signal. I'll make a separate list that goes from the peak
				#of the corrected charge signal and steps backwards in time to find the treshold
				#crossing. If none is found, it sets the timing variable to none.
				self.analysis_quantities['Passes Threshold'] = False
				if charge_energy > self.conf['Strip Threshold [sigma]']*self.baseline_rms: 
					self.analysis_quantities['Passes Threshold'] = True
					max_idx = np.argmax(self.corrected_data)
					#reverse the wave starting with the peak sample. 
					trunc_reversed_wave = self.corrected_data[:max_idx+1][::-1]
					t10 = np.where(trunc_reversed_wave < 0.1*charge_energy)[0]
					if(len(t10) == 0): t10 = None
					t25 = np.where(trunc_reversed_wave < 0.25*charge_energy)[0] 
					if(len(t25) == 0): t25 = None
					t50 = np.where(trunc_reversed_wave < 0.5*charge_energy)[0] 
					if(len(t50) == 0): t50 = None
					t90 = np.where(trunc_reversed_wave < 0.9*charge_energy)[0] 
					if(len(t90) == 0): t90 = None

					# Compute drift time in microseconds, provided these indices
					drift_time = (t90 - self.conf['Charge Pretrigger Length [samples]']) * (self.conf['Sampling Period [ns]'] / 1.e3)
                
				#only now we apply calibration constant to charge energy
				cal = self.analysis_config.GetCalibrationConstantForSoftwareChannel(self.sw_ch)
				self.analysis_quantities['Baseline'] = self.baseline
				self.analysis_quantities['Baseline RMS'] = self.baseline_rms
				self.analysis_quantities['Charge Energy'] = charge_energy*cal
				self.analysis_quantities['T10'] = t10
				self.analysis_quantities['T25'] = t25
				self.analysis_quantities['T50'] = t50
				self.analysis_quantities['T90'] = t90
				self.analysis_quantities['Drift Time'] = drift_time
				

				#If you are using induced charge in an analysis, please
				#take this moment to convert this to a run parameter instead
				#of a fixed magic number! And double check this inudction_charge
				#function; it may not have been vetted in years. 
				induction_window_ns = 4000
				ind_window_sample = int(induction_window_ns/self.sampling_period_ns)
				self.analysis_quantities['Induced Charge'] = self.Induction_Charge(ind_window_sample)
				####################################### FINISHES HERE
			else:
				pulse_area = 0.
				pulse_time = 0.
				pulse_height = 0.


		else:
			print('WARNING: the not-fixed-trigger analysis has not been tested, and may give' + \
				' spurious results.')
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



	#######################################################################################
	def GetPulseArea( self, dat_array ):
		if len(dat_array) == 0: return 0,0,0
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


	#######################################################################################
	def GetPulseAreaAndTimingParameters( self, dat_array):
		if len(dat_array) == 0: return None, None, None, None, None, None
		#data is passed in already baseline subtracted
		if 'SiPM' in self.detector_type:
			pulse_height = np.max(dat_array)
			integ_begin = self.conf['SiPM Integral Start [samples]']
			integ_end = self.conf['SiPM Integral End [samples]']
			pulse_area = np.trapz(dat_array[integ_begin:integ_end], dx=self.conf['Sampling Period [ns]'])

			#can find t10, t90 relative to the area; i.e. points in 
			#time where the area is 90% of max. 
			cumul_pulse = np.cumsum(dat_array[integ_begin:integ_end])
			thresh = np.max(cumul_pulse)
			t5 = np.where( cumul_pulse > 0.05*thresh )[0]
			if(len(t5) == 0):
				t5 = None
			else:
				t5 = t5[0]
			t10 = np.where( cumul_pulse > 0.1*thresh )[0]
			if(len(t10) == 0):
				t10 = None
			else:
				t10 = t10[0]
			t25 = np.where( cumul_pulse > 0.25*thresh )[0]
			if(len(t25) == 0):
				t25 = None
			else:
				t25 = t25[0]
			t50 = np.where( cumul_pulse > 0.50*thresh )[0]
			if(len(t50) == 0):
				t50 = None
			else:
				t50 = t50[0]
			t90 = np.where( cumul_pulse > 0.90*thresh )[0]
			if(len(t90) == 0):
				t90 = None
			else:
				t90 = t90[0]
		else:
			return None, None, None, None, None, None
		return pulse_area, pulse_height, t5, t10, t25, t50, t90



	#######################################################################################
	def Induction_Charge( self, induction_window_sample ):
		buffer_array = np.zeros(len(self.data))
		above_rms = np.where(self.data-self.analysis_quantities['Baseline']>\
				     (self.analysis_quantities['Charge Energy']+\
					3*self.analysis_quantities['Baseline RMS']))[0]
		buffer_array[above_rms] = 1

		if len(self.data)%induction_window_sample != 0:
			padding_number = induction_window_sample - len(self.data)%induction_window_sample
			buffer_array = np.pad(buffer_array,(0,padding_number),'constant', constant_values=0)

		tag = np.sum(buffer_array.reshape(-1,induction_window_sample),axis=1)
		induction_tag = np.where(tag>0.8*induction_window_sample)[0]

		if induction_tag.shape[0] == 0:
			return 0.0
		else:
			lower_bin = (induction_tag[-1]-1)*induction_window_sample
			higher_bin = (induction_tag[-1]+1)*induction_window_sample

		average_charge_ind = sum((self.data-self.analysis_quantities['Baseline'])\
					  [lower_bin:higher_bin])/(higher_bin-lower_bin)

		if average_charge_ind<30:
			return 0.0
		else:
			return average_charge_ind


############################ End the Waveform class ##############################################



##################################################################################################
# The decay time correction is recursive, and runs much faster using the just-in-time capabilities
# from numba. But, I had to make it an external function
##################################################################################################
@jit("float64[:](float64[:],float64,float64)",nopython=True)
def DecayTimeCorrection( input_wfm, decay_time_us, sampling_period_ns ):

		# Here I'll assume the decay time is in units of mircoseconds
		# and the sampling period is in units of ns
		new_wfm = np.copy( input_wfm )
		for i in range(len(input_wfm)-1):
			new_wfm[i+1] = new_wfm[i] - \
					np.exp( - (sampling_period_ns/1.e3) / decay_time_us ) * input_wfm[i] + \
					input_wfm[i+1]
		return new_wfm

@jit("float64[:](float64[:],float64)",nopython=True)
def Differentiator( wfm, decay ):
	# decay parameter is in number of samples
	z = np.exp(-1/decay)
	a0 = (1 + z)/2.
	a1 = -(1+z)/2.
	b1 = z
	out = [0]
	for i in range(1,len(wfm)):
		out.append( a0 * wfm[i] + a1 * wfm[i-1] + b1 * out[i-1])
	return np.array(out)


class Event:
	#One significant change to this class has been made recently (July 2022). 
	#This event now takes in the binary file object, an NGMBinaryFile as opposed
	#to filenames, and leaves it up to the notebook or script to determine which
	#binary file should be indexed for a particular event. This is because the output_dataframes
	#in a reduced dataset contain a column for the filename, so each event can be indexed to a binary
	#file and then the list of events to plot may be sorted by their filename - allowing one to
	#load one binary file, plot many events, then load another binary file, plot many events; as opposed
	#to loading the same binary file over and over again for each event.
	def __init__( self, bin_file, event_number, analysis_config):

		from StanfordTPCAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration
		from StanfordTPCAnalysis.ParseStruck import NGMBinaryFile
		
		if(bin_file is None):
			print("Please use your plotting script or notebook to pass a binary file, this function received None")
			return None 

		channel_number = analysis_config.GetNumberOfChannels()
		self.event_number		= event_number
		self.waveform			= {}
		self.baseline			= []
		self.charge_energy_ch		= [] #these were taken out, as the reduced file is no longer involved in this function; can be put back in
		self.risetime			= [] #these were taken out, as the reduced file is no longer involved in this function; can be put back in
		self.sampling_frequency = analysis_config.run_parameters['Sampling Rate [MHz]']

		#Read the binary file and get spills from it. 

		waveforms_from_binary = bin_file.getEventFromReducedIndex(event_number) #Extracts just that event's data from the binary file. 
		self.ix_channel = []
		#looping through channels and fill the waveforms
		for i, ch_waveform in waveforms_from_binary.items():
			ch_type = analysis_config.GetChannelTypeForSoftwareChannel(i)
			ch_name = analysis_config.GetChannelNameForSoftwareChannel(i)
			if ch_name == 'Off':
				continue
			self.ix_channel.append(i)
			self.waveform[ch_name] = Waveform(input_data = ch_waveform,\
							detector_type	    = ch_type,\
							sampling_period_ns  = 1.e3/self.sampling_frequency,\
							input_baseline	    = -1,\
							fixed_trigger	    = False,\
							trigger_position    = analysis_config.run_parameters['Pretrigger Length [samples]'],\
							decay_time_us	    = analysis_config.GetDecayTimeForSoftwareChannel( i),\
							  calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel(i),\
							strip_threshold = analysis_config.run_parameters['Strip Threshold [sigma]'])
			#same as for Waveform class
			self.baseline.append(np.mean(ch_waveform[:int(analysis_config.run_parameters['Baseline Length [samples]'])]))



	#smoothing function, the waveform is overwritten, time_width is in us
	def smooth( self, time_width ):
		for k,v in self.waveform.items():
			self.waveform[k].data = gaussian_filter( v.data.astype(float), time_width*self.sampling_frequency)
		return self.waveform


	def plot_event( self, risetime=False ):
		import matplotlib.pyplot as plt
		ch_offset = 250
		for i,e in enumerate(np.argsort(self.ix_channel)):
			v = list(self.waveform.keys())[e]
			p = plt.plot(np.arange(len(self.waveform[v].data))/self.sampling_frequency,self.waveform[v].data-self.baseline[e]+ch_offset*i)
			
		plt.xlabel('time [$\mu$s]')
		plt.tight_layout()
		return(plt)

class Simulated_Event:

	def __init__( self, reduced, path_to_tier1, event_number,\
			run_parameters_file,\
			calibrations_file,\
			channel_map_file,\
			add_noise=True):

		from StanfordTPCAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration
		from StanfordTPCAnalysis.ParseSimulation import NEXOOfflineFile
		import pickle

		try :
			if path_to_tier1[-1] != '/':
				path_to_tier1 += '/'
		except TypeError:
			pass

		analysis_config = StruckAnalysisConfiguration.StruckAnalysisConfiguration()
		analysis_config.GetRunParametersFromFile( run_parameters_file )
		analysis_config.GetCalibrationConstantsFromFile( calibrations_file )
		analysis_config.GetChannelMapFromFile( channel_map_file )
		channel_number = analysis_config.GetNumberOfChannels()
		self.event_number		= event_number
		self.waveform			= {}
		self.baseline			= []
		self.baseline_rms		= []
		self.charge_energy_ch		= []
		self.risetime			= []
		self.sampling_frequency = analysis_config.run_parameters['Simulation Sampling Rate [MHz]']

		if path_to_tier1 is not None:
			path_to_file		= path_to_tier1
			entry_from_reduced	= pd.read_hdf(reduced, start=self.event_number, stop=self.event_number+1)
			timestamp		= entry_from_reduced['Timestamp'].values[0]
			fname			= entry_from_reduced['File'].values[0]
			self.tot_charge_energy	= entry_from_reduced['TotalTileEnergy'].values[0]
			self.event_number	= entry_from_reduced['Event'][event_number]

		else:
			print('No reduced file found, charge energy and risetime information not present')
			fname = reduced.split('/')[-1]
			path_to_file = reduced[:-len(fname)]
			self.tot_charge_energy = 0.0

		pickled_fname = path_to_file + 'channel_status.p'
		global ch_status
		with open(pickled_fname,'rb') as f:
			ch_status = pickle.load(f)

		input_file = NEXOOfflineFile.NEXOOfflineFile( input_filename = path_to_file+fname,\
								config = analysis_config,\
								add_noise = add_noise,\
								noise_lib_directory='/usr/workspace/nexo/jacopod/noise/')
		if path_to_tier1 is not None:
			input_file.global_noise_file_counter = entry_from_reduced['NoiseIndex'].iloc[0][0]
			input_file.noise_file_event_counter  = entry_from_reduced['NoiseIndex'].iloc[0][1]
		input_df = input_file.GroupEventsAndWriteToHDF5(save = False, start_stop=[self.event_number,self.event_number+1])
		#since the timestamps are not filled in the simulated data there is no real handle to cross-checked the event is actually the same

		waveform = input_df['Data'][0]
		#looping through channels and fill the waveforms
		for i,ch_waveform in enumerate(waveform):
			ch_type = analysis_config.GetChannelTypeForSoftwareChannel(i)
			ch_name = analysis_config.GetChannelNameForSoftwareChannel(i)

			if ch_name in ch_status.keys():
				mean,sigma = ch_status[ch_name]
				ch_waveform = np.random.normal(mean,sigma,len(ch_waveform))

			self.waveform[ch_name] = Waveform(input_data = ch_waveform,\
							detector_type	    = ch_type,\
							sampling_period_ns  = 1.e3/self.sampling_frequency,\
							input_baseline	    = -1,\
							polarity	    = -1,\
							fixed_trigger	    = False,\
							trigger_position    = analysis_config.run_parameters['Pretrigger Length [samples]'],\
							decay_time_us	    = analysis_config.GetDecayTimeForSoftwareChannel( i ),\
							calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel(i))
			#same as for Waveform class
			self.baseline.append(np.mean(ch_waveform[:analysis_config.run_parameters['Baseline Length [samples]']]))
			#different cases for tile/SiPM
			try:
				self.charge_energy_ch.append(entry_from_reduced['{} {} Charge Energy'.format(ch_type,ch_name)].values[0])
				self.baseline_rms.append(entry_from_reduced['{} {} Baseline RMS'.format(ch_type,ch_name)].values[0])
				self.risetime.append(entry_from_reduced['{} {} T90'.format(ch_type,ch_name)].values[0]/self.sampling_frequency)
			except (KeyError, UnboundLocalError):
				self.charge_energy_ch.append(0)
				self.baseline_rms.append(0)
				self.risetime.append(0)


	#smoothing function, the waveform is overwritten, time_width is in us
	def smooth( self, time_width ):
		for k,v in self.waveform.items():
			self.waveform[k].data = gaussian_filter( v.data.astype(float), time_width*self.sampling_frequency)
		return self.waveform


	def plot_event( self, risetime=False, energy_threshold = True ):
		import matplotlib.pyplot as plt
		ch_offset = 250
		for i,v in enumerate(self.waveform):
			p = plt.plot(np.arange(len(self.waveform[v].data))/self.sampling_frequency,self.waveform[v].data-self.baseline[i]+ch_offset*i)
			if energy_threshold and self.charge_energy_ch[i]<5*self.baseline_rms[i]:
				plt.text(0,ch_offset*i,'{} 0'.format(v))
			else:
				plt.text(0,ch_offset*i,'{} {:.1f}'.format(v,self.charge_energy_ch[i]))
			if risetime and self.charge_energy_ch[i]>0:
				plt.vlines(self.risetime[i],ch_offset*i,ch_offset*i+2*self.charge_energy_ch[i],linestyles='dashed',colors=p[0].get_color())

		plt.xlabel('time [$\mu$s]')
		plt.title('Event {}, Energy {:.1f} ADC counts'.format(int(self.event_number),self.tot_charge_energy))
		plt.tight_layout()
		return(plt)

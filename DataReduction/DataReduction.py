import pandas as pd
import numpy as np
import time
from TMSAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration

from TMSAnalysis.WaveformAnalysis import Waveform

def ReduceH5File( filename, output_dir, run_parameters_file, calibrations_file, channel_map_file, \
			num_events=-1, input_baseline=-1, input_baseline_rms=-1, \
			fixed_trigger=False, fit_pulse_flag=False):

	start_time = time.time()
	filetitle = filename.split('/')[-1]
	filetitle_noext = filetitle.split('.')[0]
	outputfile = '{}_reduced.h5'.format(filetitle_noext)

	analysis_config = StruckAnalysisConfiguration.StruckAnalysisConfiguration()
	analysis_config.GetRunParametersFromFile( run_parameters_file )
	analysis_config.GetCalibrationConstantsFromFile( calibrations_file )
	analysis_config.GetChannelMapFromFile( channel_map_file )
	sampling_period_ns = 1./(analysis_config.run_parameters['Sampling Rate [MHz]']/1.e3)

	output_series = pd.Series()
	output_df = pd.DataFrame()

	input_df = pd.read_hdf(filename)
	input_columns = input_df.columns
	output_columns = [col for col in input_columns if (col!='Data') and (col!='Channels')]


	event_counter = 0

	for index, thisrow in input_df.iterrows():
		if (event_counter > num_events) and (num_events > 0):
			break
		if event_counter % 50 == 0:
			print('\tProcessing event {} at {:4.4}s...'.format(event_counter,time.time()-start_time))
		# Set all the values that are not the waveform/channel values
		for col in output_columns:
			output_series[col] = thisrow[col]

		output_series['TotalTileEnergy'] = 0.
		output_series['TotalSiPMEnergy'] = 0.
		output_series['NumTileChannelsHit'] = 0
		output_series['NumXTileChannelsHit'] = 0
		output_series['NumYTileChannelsHit'] = 0
		output_series['NumSiPMChannelsHit'] = 0
		output_series['TimeOfMaxChannel'] = 0
		max_channel_val = 0.
		# Loop through channels, do the analysis, put this into the output series
		for ch_num in range(len(thisrow['Channels'])):
			software_ch_num = thisrow['Channels'][ch_num]
			#calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel( software_ch_num )
			#decay_constant = analysis_config.GetDecayTimeForSoftwareChannel( software_ch_num )
			polarity = -1.
			if analysis_config.run_parameters['Sampling Rate [MHz]'] == 62.5:
				polarity = 1.
			w = Waveform.Waveform(input_data=thisrow['Data'][ch_num],\
						detector_type       = thisrow['ChannelTypes'][ch_num],\
						sampling_period_ns  = sampling_period_ns,\
						input_baseline      = input_baseline,\
						input_baseline_rms  = input_baseline_rms,\
						polarity            = polarity,\
						fixed_trigger       = fixed_trigger,\
						trigger_position    = analysis_config.run_parameters['Pretrigger Length [samples]'],\
						decay_time_us       = analysis_config.GetDecayTimeForSoftwareChannel( software_ch_num ),\
						calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel( software_ch_num ) )
			w.FindPulsesAndComputeAQs(fit_pulse_flag=fit_pulse_flag)
			for key in w.analysis_quantities.keys():
				output_series['{} {} {}'.format(\
								analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ),\
								analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ),\
								key)] = w.analysis_quantities[key]
			if 'TileStrip' in analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ):
				if w.analysis_quantities['Charge Energy'] > 0.:
					output_series['NumTileChannelsHit'] += 1
					if 'X' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ):
						output_series['NumXTileChannelsHit'] += 1
					if 'Y' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ):
						output_series['NumYTileChannelsHit'] += 1
					output_series['TotalTileEnergy'] += w.analysis_quantities['Charge Energy']
					if w.analysis_quantities['Charge Energy']**2 > max_channel_val**2:
						max_channel_val = w.analysis_quantities['Charge Energy']
						output_series['TimeOfMaxChannel'] = w.analysis_quantities['T90']
			if 'SiPM' in analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ):
				if w.analysis_quantities['Pulse Area'] > 0.:
					output_series['NumSiPMChannelsHit'] += 1
					output_series['TotalSiPMEnergy'] += w.analysis_quantities['Pulse Area']
		# Append this event to the output dataframe
		output_series['File'] = filetitle
		output_series['Event'] = event_counter
		output_df = output_df.append(output_series,ignore_index=True)
		event_counter += 1
	output_df.to_hdf(output_dir + outputfile,key='df')	
	print('Run time: {:4.4}'.format(time.time()-start_time))

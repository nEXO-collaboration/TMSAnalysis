import pandas as pd
import numpy as np
import time

from TMSAnalysis.WaveformAnalysis import Waveform

def ReduceH5File( filename, output_dir, num_events=-1, sampling_period=16., input_baseline=-1, input_baseline_rms=-1, polarity=-1., \
			fixed_trigger=False, trigger_position=0, fit_pulse_flag=False):

	start_time = time.time()
	filetitle = filename.split('/')[-1]
	filetitle_noext = filetitle.split('.')[0]
	outputfile = '{}_reduced.h5'.format(filetitle_noext)

	output_series = pd.Series()
	output_df = pd.DataFrame()

	input_df = pd.read_hdf(filename)
	input_columns = input_df.columns
	print(input_columns)
	output_columns = [col for col in input_columns if (col!='Data') and (col!='Channels')]

	event_counter = 0
	for index, thisrow in input_df.iterrows():
		if (event_counter > num_events) and (num_events > 0):
			break
		if event_counter % 500 == 0:
			print('Processing event {}...'.format(event_counter))
		# Set all the values that are not the waveform/channel values
		for col in output_columns:
			output_series[col] = thisrow[col]

		output_series['TotalEnergy'] = 0.
		output_series['NumChannelsHit'] = 0
		# Loop through channels, do the analysis, put this into the output series
		for ch_num in range(len(thisrow['Channels'])):
			w = Waveform.Waveform(input_data=thisrow['Data'][ch_num],\
						detector_type=thisrow['ChannelTypes'][ch_num],\
						sampling_period=sampling_period,\
						input_baseline=input_baseline,\
						input_baseline_rms=input_baseline_rms,\
						polarity=polarity,\
						fixed_trigger=fixed_trigger,\
						trigger_position=trigger_position)
			w.FindPulsesAndComputeArea(fit_pulse_flag=fit_pulse_flag)
			for key in w.analysis_quantities.keys():
				output_series['{}{} {}'.format(\
								thisrow['ChannelTypes'][ch_num],\
								thisrow['ChannelPositions'][ch_num],\
								key)] = w.analysis_quantities[key]
			if w.analysis_quantities['Pulse Areas'] > 0. and ('TileStrip' in thisrow['ChannelTypes']):
				output_series['NumTileChannelsHit'] += 1
				output_series['TotalTileEnergy'] += w.analysis_quantities['Pulse Areas']
		# Append this event to the output dataframe
		output_series['File'] = filetitle
		output_series['Event'] = event_counter
		output_df = output_df.append(output_series,ignore_index=True)
		event_counter += 1
	output_df.to_hdf(output_dir + outputfile,key='df')	
	print('Run time: {:4.4}'.format(time.time()-start_time))

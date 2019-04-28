import pandas as pd
import numpy as np

from TMSAnalysis.WaveformAnalysis import Waveform

def ReduceH5File( filename, num_events=-1 ):

	filetitle = filename.split('/')[-1]
	filetitle_noext = filetitle.split('.')[0]
	outputfile = '{}_reduced.h5'.format(filetitle_noext)

	output_series = pd.Series()
	output_df = pd.DataFrame()

	input_df = pd.read_hdf(filename)
	input_columns = input_df.columns
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
		# Loop through channels, do the analysis, put this into the output series
		for ch_num in thisrow['Channels']:
			w = Waveform.Waveform(input_data=thisrow['Data'][ch_num],\
						detector_type=thisrow['DetectorType'][ch_num],\
						sampling_rate=10.)
			w.FindPulsesAndComputeArea()
			for key in w.analysis_quantities.keys():
				output_series['Ch{} {}'.format(ch_num,key)] = w.analysis_quantities[key]
		# Append this event to the output dataframe
		output_df = output_df.append(output_series,ignore_index=True)
		event_counter += 1
	output_df.to_hdf(outputfile,key='df')	

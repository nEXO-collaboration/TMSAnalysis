import pandas as pd
import numpy as np
from .WavedumpFile import WavedumpFile
import os

class WavedumpDataset:
		
        ####################################################################
	def __init__( self, datapath=None, datasetID=None ):
		print('WavedumpDataset object constructed.')
		self.datapath = datapath
		self.datasetID = datasetID
		self.wavedumpFiles = None
		
	####################################################################
	def LoadFiles( self ):
		if self.datapath is None:
			raise Exception('No data path given.')
		if self.datasetID is None:
			raise Exception('No datasetID given.')
		if not os.path.isdir(self.datapath):
   			raise Exception('Data path is not a directory.')
		allfiles = os.listdir(self.datapath)	
		print('All files: {}'.format(allfiles))
		relevantfiles = [f for f in allfiles if (self.datasetID in f) and (f.endswith('txt'))]
		print('Relevant files: {}'.format(relevantfiles))
		self.wavedumpFiles = [WavedumpFile('{}/{}'.format(self.datapath,f)) for f in relevantfiles]
		self.num_channels = len(self.wavedumpFiles)
		print('Num channels loaded: {}',self.num_channels)

        ####################################################################
	def ConvertFilesToHDF5( self, num_events=-1 ):
		if self.wavedumpFiles is None:
			self.LoadFiles()
		
		output_df = pd.DataFrame(columns=['Record Length','BoardID','Channels',\
						'Event Number','Pattern','Trigger Time Stamp',\
						'DC Offset (DAC)','Data'])
		output_row = pd.Series()
		event_counter = 0

		while True:
			if (event_counter > num_events) and (num_events > 0):
				break
			try:
				for f in self.wavedumpFiles:
					f.ReadEvent()
			except:
				break

			waveform_data = [f.current_evt['Data'] for f in self.wavedumpFiles]
			channels = [f.current_evt['Channel'] for f in self.wavedumpFiles]

			output_row['Record Length'] = self.wavedumpFiles[0].current_evt['Record Length'] 
			output_row['BoardID'] = self.wavedumpFiles[0].current_evt['BoardID']
			output_row['Channels'] = channels
			output_row['Event Number'] = self.wavedumpFiles[0].current_evt['Event Number']
			output_row['Pattern'] = self.wavedumpFiles[0].current_evt['Pattern']
			output_row['Trigger Time Stamp'] = self.wavedumpFiles[0].current_evt['Trigger Time Stamp']
			output_row['DC Offset (DAC)'] = self.wavedumpFiles[0].current_evt['DC Offset (DAC)']
			output_row['Data'] = waveform_data
			
			output_df = output_df.append(output_row,ignore_index=True)
			event_counter += 1

		output_filename = '{}.h5'.format(self.datasetID)
		output_df.to_hdf(output_filename,key='df')



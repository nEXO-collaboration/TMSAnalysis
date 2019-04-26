############################################################################
# This file defines a class that reads the raw ASCII files from
# CAEN's wavedump software and puts them into Pandas dataframes 
# for analysis purposes.
#
#    - Brian L.
############################################################################

import pandas as pd
import numpy as np


class WavedumpFile:

        ####################################################################
	def __init__( self, filename=None ):
		print('WavedumpFile object constructed.')
		if filename is not None:
			self.LoadFile( filename )

        ####################################################################
	def LoadFile( self, filename ):
		self.infile = open(filename,'r')
		print('Input file: {}'.format(self.infile.name))

        ####################################################################
	def ReadEvent( self ):
		try:
			self.infile
		except NameError:
			self.LoadFile()
			
		self.current_evt = pd.Series()
		try:
			self.current_evt['Record Length'] = self.infile.readline().split()[-1]
		except IndexError:
			raise Exception('End of file.')

		self.current_evt['BoardID'] = self.infile.readline().split()[-1]
		self.current_evt['Channel'] = int(self.infile.readline().split()[-1])
		self.current_evt['Event Number'] = int(self.infile.readline().split()[-1])
		if self.current_evt['Event Number'] % 500 == 0:
			print('Processing event {}...'.format(self.current_evt['Event Number']))
		self.current_evt['Pattern'] = self.infile.readline().split()[-1]
		self.current_evt['Trigger Time Stamp'] = self.infile.readline().split()[-1]
		self.current_evt['DC Offset (DAC)'] = self.infile.readline().split()[-1]

		dat_array = np.array([])
		# Loop to add data to a numpy array
		for i in range(0,int(self.current_evt['Record Length'])):
			lineptr = self.infile.tell()
			thisline = self.infile.readline()
			if not thisline:
				break
			try:
				dat_pt = float(thisline)
				dat_array = np.append(dat_array,dat_pt)
			except ValueError:
				print('Waveform truncated.')
				self.infile.seek(lineptr)
				print('Event {}'.format(self.current_evt['Event Number']))
				break
		self.current_evt['Data'] = dat_array

		return self.current_evt
	
        ####################################################################
	def ConvertFileToHDF5( self ):
		try:
			self.infile
		except NameError:
			self.LoadFile()
		
		output_df = pd.DataFrame(columns=['Record Length','BoardID','Channel',\
						'Event Number','Pattern','Trigger Time Stamp',\
						'DC Offset (DAC)','Data'])
		while True:
			try:
				self.ReadEvent()
			except:
				break
			output_df = output_df.append(self.current_evt,ignore_index=True)
		
		output_filename = '{}.h5'.format(self.GetFileTitle(self.infile.name))
		output_df.to_hdf(output_filename,key='raw')
		
        ####################################################################
	def GetFileTitle( self, filepath ):
		filename = filepath.split('/')[-1]
		filetitle = filename.split('.')[0]
		return filetitle
			

############################################################################
# This file defines a class that reads the ROOT files produced by the 
# NGM Daq and puts the data into Pandas dataframes 
# for analysis purposes.
#
#    - Brian L.
############################################################################

import pandas as pd
import numpy as np
import uproot as up


class NGMRootFile:

        ####################################################################
	def __init__( self, filename=None ):
		print('NGMFile object constructed.')
		if filename is not None:
			self.LoadFile( filename )

        ####################################################################
	def LoadFile( self, filename ):
		self.infile = up.open(filename)
		print('Input file: {}'.format(self.infile.name))
		try:
			self.intree = self.infile['HitTree'].pandas.df()
		except:
			print('Some problem getting the HitTree out of the file.')
			self.infile.close()
			return
		self.infile.close()
		print('Got HitTree.')
		

        ####################################################################
	def GroupEventsAndWriteToHDF5( self, nevents = -1 ):
		
		try:
			self.infile
		except NameError:
			self.LoadFile()
			
		self.current_evt = pd.Series()
		self.outputdf = pd.DataFrame()
		this_event_timestamp = -1
		channels = []
		data = []
		counter = 0

		for index,thisrow in self.intree.iterrows():
			if nevents > 0:
				if counter > nevents:
					break
			# If the timestamp has changed (and it's not the first line), write the output
			# to the output dataframe.
			if (index not this_event_timestamp) and (this_event_index > 0):
				self.current_evt['Channels'] = channels
				self.current_evt['Data'] = data
				self.outputdf = self.outputdf.append( self.current_evt, ignore_index=True )
				channels = []
				data = []
			else:
				self.current_evt['Timestamp'] = thisrow['_rawclock']
				channels.append( thisrow['_channel'] + thisrow['_slot']*16 )
				data.append( thisrow['_waveform'] )
			counter += 1

		output_filename = '{}.h5'.format(self.GetFileTitle(self.infile.name))
		self.outputdf.to_hdf(output_filename,key='raw')
	
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
			

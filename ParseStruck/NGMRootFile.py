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
import sys
import time


class NGMRootFile:

        ####################################################################
	def __init__( self, filename=None ):
		print('NGMFile object constructed.')
		if filename is not None:
			self.LoadFile( filename )
		self.h5_file = None

        ####################################################################
	def LoadRootFile( self, filename ):
		self.infile = up.open(filename)
		self.filename = filename
		print('Input file: {}'.format(self.filename))
		try:
			self.intree = self.infile['HitTree'].pandas.df(flatten=False)
		except ValueError as e:
			print('Some problem getting the HitTree out of the file.')
			print('{}'.format(e))
			return
		print('Got HitTree.')
		

        ####################################################################
	def GroupEventsAndWriteToHDF5( self ):
		
		try:
			self.infile
		except NameError:
			self.LoadFile()
	
		start_time = time.time()		
		self.current_evt = pd.Series()
		self.outputdf = pd.DataFrame(columns=['Timestamp','Channels','Data'])
		this_event_timestamp = -1
		channels = []
		data = []

		for index,thisrow in self.intree.iterrows():
			# If the timestamp has changed (and it's not the first line), write the output
			# to the output dataframe.
			if (thisrow['_rawclock'] != this_event_timestamp) and (this_event_timestamp > 0):
				self.current_evt['Channels'] = channels
				self.current_evt['Data'] = data
				self.outputdf = self.outputdf.append( self.current_evt, ignore_index=True )
				channels = []
				data = []
			self.current_evt['Timestamp'] = thisrow['_rawclock']
			channels.append( thisrow['_channel'] + thisrow['_slot']*16 )
			data.append( thisrow['_waveform'] )
			this_event_timestamp = thisrow['_rawclock']
		self.current_evt['Channels'] = channels
		self.current_evt['Data'] = data
		self.outputdf = self.outputdf.append( self.current_evt, ignore_index=True )

		output_filename = '{}.h5'.format(self.GetFileTitle(self.filename))
		self.outputdf.to_hdf(output_filename,key='raw')
		end_time=time.time()
		print('{:.2} seconds elapsed. {:3.3} seconds per event.'.format( \
					end_time-start_time,\
					float( (end_time-start_time)/len(self.outputdf) ) ) )
		self.h5_file = pd.read_hdf(output_filename)
		
        ####################################################################
	def GetFileTitle( self, filepath ):
		filename = filepath.split('/')[-1]
		filetitle = filename.split('.')[0]
		return filetitle
		
	
	

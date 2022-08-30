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
import time
import os
import sys


class NGMRootFile:

        ####################################################################
	def __init__( self, input_filename=None, output_directory=None, config=None, start_stop = [None, None]):
		print('NGMFile object constructed.')

		self.start_stop = start_stop
		package_directory = os.path.dirname(os.path.abspath(__file__))
		if output_directory is None:
			self.output_directory = './'
		else:
			self.output_directory = output_directory + '/'

		if input_filename is not None:
			self.LoadRootFile( input_filename )
		if config is not None:
			self.channel_map = config.channel_map
		else:
			print('WARNING: No channel map file provided. Using the default one...')
			self.channel_map = pd.read_csv(package_directory + '/channel_map_8ns_sampling.txt',skiprows=9)
		self.h5_file = None

        ####################################################################
	def LoadRootFile( self, filename ):
		self.infile = up.open(filename)
		self.filename = filename
		print('Input file: {}'.format(self.filename))
		try:
			self.intree = self.infile['HitTree']
		except ValueError as e:
			print('Some problem getting the HitTree out of the file.')
			print('{}'.format(e))
			return
		print('Got HitTree.')
		

        ####################################################################
	def GroupEventsAndWriteToHDF5( self, nevents = -1, save = True, start_stop = None, file_counter = 0 ):
		
		try:
			self.infile
		except NameError:
			self.LoadFile()

		if start_stop is not None:
			self.start_stop = start_stop	

		start_time = time.time()		
		global_evt_counter = 0
		local_evt_counter = 0
		df = pd.DataFrame(columns=['Channels','Timestamp','Data','ChannelTypes','ChannelPositions'])
		start_time = time.time()
		print('{} entries per event.'.format(len(self.channel_map)))

		# Grabs data in chunks; the chunk size corresponds to the number of active channels in the acquisition.
		for data in self.intree.iterate(['_waveform','_rawclock','_slot','_channel'],\
						step_size=len(self.channel_map),\
						entry_start=self.start_stop[0],\
						entry_stop=self.start_stop[1]):
			if nevents > 0:
				if global_evt_counter > nevents:
					break
			data_series = pd.Series(data)
			channel_mask, channel_types, channel_positions = self.GenerateChannelMask( data['_slot'],data['_channel'])
                         
                        # Remove 'Off' channels from the data stream, uproot v3
			#for column in data_series.items():
			#	data_series[ column[0] ] = np.array(data_series[column[0]][channel_mask])
			######## uproot4 compatiblility: NOT TESTED YET ####### 2022/08/15 Jacopo
			data = data[channel_mask]
			#######################################################
			output_series = pd.Series()
			output_series['Channels'] = data_series['_slot']*16+data_series['_channel']
			output_series['Timestamp'] = data_series['_rawclock']
			output_series['Data'] = data_series['_waveform']
			output_series['ChannelTypes'] = np.array(channel_types)[channel_mask]
			output_series['ChannelPositions'] = channel_positions[channel_mask]
			df = df.append(output_series,ignore_index=True)	


			global_evt_counter += 1
			local_evt_counter += 1
#			if local_evt_counter > 200 and save:
#				output_filename = '{}{}_{:0>3}.h5'.format( self.output_directory,\
#									self.GetFileTitle(str(self.infile.name)),\
#									file_counter )
#				df.to_hdf(output_filename,key='raw')
#				local_evt_counter = 0
#				file_counter += 1
#				df = pd.DataFrame(columns=['Channels','Timestamp','Data','ChannelTypes','ChannelPositions'])
#				print('Written to {} at {:4.4} seconds'.format(output_filename,time.time()-start_time))	
		

		if save:
			output_filename = '{}{}_{:0>3}.h5'.format( self.output_directory,\
								self.GetFileTitle(str(self.infile.name)),\
								file_counter )
			df.to_hdf(output_filename,key='raw')
			end_time = time.time()
			print('{} events written in {:4.4} seconds.'.format(global_evt_counter,end_time-start_time))
			return df
		else:
			return df
	
	####################################################################
	def GenerateChannelMask( self, slot_column, channel_column ):

		channel_mask = np.array(np.ones(len(slot_column),dtype=bool))
		channel_types = ['' for i in range(len(slot_column))]
		channel_positions = np.zeros(len(slot_column),dtype=int)

		for index,row in self.channel_map.iterrows():
			
			### Get the index of the correct slot/chan pair in the data columns
			slot_mask = np.where(slot_column==row['Board'])
			chan_mask = np.where(channel_column==row['InputChannel'])
			intersection = np.intersect1d(slot_mask,chan_mask)
			if len(intersection) == 1:
				this_index = intersection[0]
			else:
				 continue
			###
			channel_types[this_index] = row['ChannelType']
			channel_positions[this_index] = row['ChannelPosX'] if row['ChannelPosX'] != 0 else row['ChannelPosY']
			if row['ChannelType']=='Off':
				channel_mask[this_index] = False
		return channel_mask, channel_types, channel_positions
	
        ####################################################################
	def GetFileTitle( self, filepath ):
		filename = filepath.split('/')[-1]
		filetitle = filename.split('.')[0]
		return filetitle
		
	####################################################################
	def GetTotalEntries( self ):
		return self.intree.num_entries

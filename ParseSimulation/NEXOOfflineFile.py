############################################################################
# This file defines a class that reads the ROOT files produced by the 
# nexo-offline charge simulation, and converts them into a format usable by
# the TMSAnalysis software.
#
#    - Brian L.
############################################################################

import pandas as pd
import numpy as np
import uproot as up
import time
import os
import sys


class NEXOOfflineFile:

        ####################################################################
        def __init__( self, input_filename=None, output_directory=None,\
                      channel_map_file=None, start_stop = [None, None],\
                      add_noise=True, noise_lib_directory=None):

            self.start_stop = start_stop
            package_directory = os.path.dirname(os.path.abspath(__file__))
            if output_directory is None:
               self.output_directory = './'
            else:
               self.output_directory = output_directory + '/'
           
            if input_filename is not None:
               self.LoadRootFile( input_filename )
            if channel_map_file is not None:
               self.channel_map = pd.read_csv(channel_map_file)
            else:
               print('\nWARNING: No channel map file provided.')
               self.channel_map = pd.read_csv( package_directory + \
                                               '/channel_map_8ns_sampling.txt',\
                                               skiprows=9 )

            if add_noise:
               self.add_noise = True
               if noise_lib_directory is None:
                   print('\nERROR: add_noise is set to True, but no noise library is given.')
                   print('Data reduction will proceed, but waveforms will be generated with no noise.')
                   self.add_noise = False
            else:
               self.add_noise = False

            self.h5_file = None


        ####################################################################
        def LoadRootFile( self, filename ):
            self.infile = up.open(filename)
            self.filename = filename
            print('Input file: {}'.format(filename))
            try: 
               self.intree = self.infile['Event/Elec/ElecEvent']
            except ValueError as e:
               print('Some problem getting the ElecEvent tree out of the file.')
               print('{}'.format(e))
               return
            print('Finished loading.')

        ####################################################################
        def GroupEventsAndWriteToHDF5( self, nevents=-1, save=True ):
            try: 
               self.infile
            except NameError as e:
               print('\nERROR: File not properly loaded. See error message below:\n');
               print(e)

            start_time = time.time()
            global_evt_counter = 0
            local_evt_counter = 0
            file_counter = 0
            df = pd.DataFrame(columns=['Channels',\
                                       'Timestamp',\
                                       'Data',\
                                       'ChannelTypes',\
                                       'ChannelPositions'])
            
            # loop over events in the ElecEvent tree
            print('Beginning data loop.')
            for data in self.intree.iterate(['fElecChannels.fWFAmplitude',\
                                             'fElecChannels.fChannelLocalId'],\
                                            namedecode='utf-8',\
                                            entrysteps=1,\
                                            entrystart=self.start_stop[0],\
                                            entrystop=self.start_stop[1]):
                print('In data loop')
                if nevents > 0:
                   if global_evt_counter > nevents:
                      break

                data_series = pd.Series(data)
                channel_ids, channel_waveforms, \
                channel_types, channel_positions = self.GroupSimChannelsIntoDataChannels( data_series)
                output_series = pd.Series()
                output_series['Channels'] = channel_ids
                output_series['Timestamp'] = np.zeros(len(channel_ids))
                output_series['Data'] = channel_waveforms
                output_series['ChannelTypes'] = channel_types
                output_series['ChannelPositions'] = channel_positions
                df = df.append(output_series,ignore_index=True)

                global_evt_counter += 1
                local_evt_counter += 1
                if local_evt_counter > 200 and save:
                   output_filename = '{}{}_{:0>3}.h5'.format( self.output_directory,\
                                                              self.GetFileTitle(str(self.infile.name)),\
                                                              file_counter )
                   df.to_hdf(output_filename,key='raw')
                   local_evt_counter = 0
                   file_counter += 1
                   df = pd.DataFrame(columns=['Channels',\
                                              'Timestamp',\
                                              'Data',\
                                              'ChannelTypes',\
                                              'ChannelPositions'])
                   print('Written to {} at {:4.4} seconds'.format(output_filename,time.time()-start_time))

            if save:
               output_filename = '{}{}_{:0>3}.h5'.format( self.output_directory,\
                                                          self.GetFileTitle(str(self.infile.name)),\
                                                          file_counter )
               print('Saving to {}'.format(output_filename))
               df.to_hdf(output_filename,key='raw')
               end_time = time.time()
               print('{} events written in {:4.4} seconds.'.format(global_evt_counter,end_time-start_time))
            else:
               return df 



        ####################################################################
        def GroupSimChannelsIntoDataChannels( self, data_series ):

            data_channels_wfm_dict = dict()
            channel_ids = []
            channel_waveforms = []
            channel_types = []
            channel_positions = []

            for index,row in self.channel_map.iterrows():

                channel_ids.append( row['SoftwareChannel'] )
                channel_types.append( row['ChannelType'] )
                channel_positions.append( row['ChannelPosX'] if row['ChannelPosX'] != 0 else row['ChannelPosY'] )

                # Below we create the ganged-together channels by summing waveforms.
                # The try/catch and if/else blocks catch SiPM channels and fill them with empty waveforms.
                try:
                   mc_channels_in_data_channel = np.array( row['ChargeMCChannelMap'].split(','), dtype=int )
                except ValueError:
                   mc_channels_in_data_channel = None

                if mc_channels_in_data_channel is None:
                   summed_wfm = np.zeros( len(data_series['fElecChannels.fWFAmplitude'][0][0]) )
                else:
                   channels_mask = np.array( [True if int(channel) in mc_channels_in_data_channel else False\
                                                for channel in data_series['fElecChannels.fChannelLocalId'][0] ] )
                
                   summed_wfm = np.sum( np.array(data_series['fElecChannels.fWFAmplitude'][0])[channels_mask], axis=0 )
                   summed_wfm = summed_wfm * 9. # This scales the waveform to units of electrons. 

                   if self.add_noise:
                      pointless_variable = 1 # Under construction

                channel_waveforms.append( summed_wfm )

            return np.array( channel_ids ), np.array( channel_waveforms ), \
                   np.array( channel_types ), np.array( channel_positions )
       

        ####################################################################
        def GetFileTitle( self, filepath ):
            filename = filepath.split('/')[-1]
            filetitle = filename.split('.')[0]
            return filetitle

############################################################################
# This file defines a class that reads the ROOT files produced by the 
# nexo-offline charge simulation, and converts them into a format usable by
# the StanfordTPCAnalysis software.
#
#    - Brian L.
############################################################################

import pandas as pd
import numpy as np
import uproot as up
import time, random
import os
import sys


class NEXOOfflineFile:

        ####################################################################
        def __init__( self, input_filename=None, output_directory=None,\
                      config=None, start_stop = [None, None],\
                      add_noise=True, noise_lib_directory=None, verbose=False):

            self.start_stop                = start_stop
            self.analysis_config           = config
            self.global_noise_file_counter = None
            self.noise_file_event_counter  = None
            self.noise_file_event_list = []

            # Since the simulations have a longer sampling period, the following allows us to 
            # get the right pre-trigger and waveform lengths.
            self.wfm_sampling_ratio = self.analysis_config.run_parameters['Sampling Rate [MHz]']\
                                       / self.analysis_config.run_parameters['Simulation Sampling Rate [MHz]']
            if self.wfm_sampling_ratio % 1 != 0:
                 print('WARNING! The data and simulation sampling rates are not exact multiples.\n'+\
                       '         This may cause issues, specifically with adding noise to the\n'+\
                       '         simulated waveforms.')
            self.sim_wfm_length = int( self.analysis_config.run_parameters['Charge Waveform Length [samples]']\
                                       / self.wfm_sampling_ratio )
            self.sim_pretrigger_length = int( self.analysis_config.run_parameters['Charge Pretrigger Length [samples]']\
                                              / self.wfm_sampling_ratio )
            self.verbose = verbose

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
               print('\nWARNING: No channel map file provided.')
               self.channel_map = pd.read_csv( package_directory + \
                                               '/channel_map_8ns_sampling.txt',\
                                               skiprows=9 )
            # Setting up the noise library
            self.current_noise_file = None
            if add_noise:
               self.SetUpNoiseLib( noise_lib_directory )
            else:
               self.add_noise = False


            self.h5_file = None

        ####################################################################
        def SetUpNoiseLib( self, noise_lib_directory ):
           if noise_lib_directory is None:
              print('\n')
              print('ERROR: add_noise is set to True, but no noise library is given.')
              print('Data reduction will proceed, but waveforms will be generated with no noise.')
              self.add_noise = False
           else:
              self.add_noise = True
              self.noise_lib_directory = noise_lib_directory
              self.noise_library_files = [noisefile for noisefile in os.listdir(noise_lib_directory)\
                                             if (noisefile.endswith('.h5') and 'noise' in noisefile)]
              if len(self.noise_library_files) == 0:
                   print('ERROR: No noise files in the noise library directory!')
                   print('Data reduction will proceed, but waveforms will be generated with no noise.')
                   self.add_noise = False
              else:
                  if self.verbose:
                      print('Noise library files (located at {}):'.format(self.noise_lib_directory))
                      for filename in self.noise_library_files:
                          print('\t{}'.format(filename))
 
#        ####################################################################
#        def GetNoiseEvent( self ):
#            if (self.global_noise_file_counter is None) and (self.noise_file_event_counter is None):
#                self.global_noise_file_counter = random.randrange(len(self.noise_library_files))
#                print('Getting noise event.')
#                print('Reading {}'.format(self.noise_library_files[ self.global_noise_file_counter ]))
#                self.current_noise_file = pd.read_hdf( self.noise_lib_directory +\
#                                                  '/' + \
#                                                   self.noise_library_files[ self.global_noise_file_counter ] )
#                print('.....Done.')
#                self.noise_file_event_counter = random.randrange(len(self.current_noise_file))
#                this_evt = self.current_noise_file.iloc[ self.noise_file_event_counter ]
#            else:
#                self.current_noise_file = pd.read_hdf( self.noise_lib_directory +\
#                                                  '/' + \
#                                                   self.noise_library_files[ self.global_noise_file_counter ] )
#                this_evt = self.current_noise_file.iloc[ self.noise_file_event_counter ]
#            return this_evt

        ####################################################################
        def GetNoiseEvent( self ):
            if type(self.current_noise_file) == type(None) or len(self.noise_file_event_list) > 75:
                self.noise_file_event_list = []
                self.global_noise_file_counter = random.randrange(len(self.noise_library_files))
                print('Loading new noise file.')
                print('Reading {}'.format(self.noise_library_files[ self.global_noise_file_counter ]))
                self.current_noise_file = pd.read_hdf( self.noise_lib_directory +\
                                                  '/' + \
                                                   self.noise_library_files[ self.global_noise_file_counter ] )
                print('.....Done.')
            event_counter_temp = random.randrange(len(self.current_noise_file))
            while event_counter_temp in self.noise_file_event_list:
                event_counter_temp = random.randrange(len(self.current_noise_file))
            self.noise_file_event_counter = event_counter_temp
           
            self.noise_file_event_list.append(self.noise_file_event_counter)
            this_evt = self.current_noise_file.iloc[ self.noise_file_event_counter ]
            return this_evt

        ####################################################################
        def LoadRootFile( self, filename ):
            self.infile = up.open(filename)
            self.filename = filename
            if self.verbose:
                print('Input file: {}'.format(filename))
            try: 
               self.electree = self.infile['Event/Elec/ElecEvent']
               self.simtree = self.infile['Event/Sim/SimEvent']
            except ValueError as e:
               print('Some problem getting the ElecEvent tree out of the file.')
               print('{}'.format(e))
               return
            print('Finished loading.')

        ####################################################################
        def GroupEventsAndWriteToHDF5( self, nevents=-1, save=True, start_stop=None ):
            try: 
               self.infile
            except NameError as e:
               print('\nERROR: File not properly loaded. See error message below:\n');
               print(e)
            if start_stop is not None:
                  self.start_stop = start_stop
            else:
                  self.start_stop = [0,self.GetTotalEntries()]

            start_time = time.time()
            global_evt_counter = 0
            local_evt_counter = 0
            file_counter = 0
            df = pd.DataFrame(columns=['Channels',\
                                       'Timestamp',\
                                       'Data',\
                                       'ChannelTypes',\
                                       'ChannelPositions',\
                                       'MCElectrons',\
                                       'MCPhotons'])
            
            # loop over events in the ElecEvent tree
            if self.verbose:
                print('Beginning data loop.')
            counter = 0
            for data in self.electree.iterate(['fElecChannels.fWFAmplitude',\
                                             'fElecChannels.fChannelLocalId'],\
                                            entry_start=self.start_stop[0],\
                                            entry_stop=self.start_stop[1],\
                                            step_size=1, library='np'):
                #print('Event {}'.format(counter))
                simdata = [thisdata for thisdata in self.simtree.iterate(['fNTE','fInitNOP','fGenX','fGenY','fGenZ'],\
                                                                         step_size=1,\
                                                                         entry_start=self.start_stop[0]+counter,\
                                                                         entry_stop=self.start_stop[0]+counter+1, library='np')][0]
                counter += 1
                if nevents > 0:
                   if global_evt_counter > nevents:
                      break
                data_series = pd.Series(data)
                simdata_series = pd.Series(simdata)
                channel_ids, channel_waveforms, \
                channel_types, channel_positions = self.GroupSimChannelsIntoDataChannels( data_series)
                output_series = {}
                output_series['Channels'] = channel_ids
                output_series['Timestamp'] = np.zeros(len(channel_ids))
                output_series['Data'] = channel_waveforms
                output_series['ChannelTypes'] = channel_types
                output_series['ChannelPositions'] = channel_positions
                output_series['NoiseIndex'] = (self.global_noise_file_counter , self.noise_file_event_counter)
                output_series['MCElectrons'] = float(simdata_series['fNTE'][0])
                output_series['MCPhotons'] = float(simdata_series['fInitNOP'][0])
                output_series['MC_GenX'] = float(simdata_series['fGenX'][0])
                output_series['MC_GenY'] = float(simdata_series['fGenY'][0])
                output_series['MC_GenZ'] = float(simdata_series['fGenZ'][0])
                self.global_noise_file_counter = None
                self.noise_file_event_counter  = None

                df = pd.concat( [df, pd.Series(output_series)],ignore_index=True)

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
                                              'ChannelPositions',\
                                              'MCElectrons',\
                                              'MCPhotons'])
                   print('Written to {} at {:4.4} seconds'.format(output_filename,time.time()-start_time))

            if save:
               output_filename = '{}{}_{:0>3}.h5'.format( self.output_directory,\
                                                          self.GetFileTitle(str(self.infile.name)),\
                                                          file_counter )
               print('Saving to {}'.format(output_filename))
               df.to_hdf(output_filename,key='raw')
               end_time = time.time()
               print('{} events written in {:4.4} seconds.'.format(global_evt_counter,end_time-start_time))
               return df
            else:
               return df 



        ####################################################################
        def GroupSimChannelsIntoDataChannels( self, data_series ):

            data_channels_wfm_dict = dict()
            channel_ids = []
            channel_waveforms = []
            channel_types = []
            channel_positions = []

            if self.add_noise:
               noise_waveforms = self.GetNoiseEvent()

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

                if mc_channels_in_data_channel is None or len(data_series['fElecChannels.fWFAmplitude'][0]) == 0:
                   summed_wfm = np.zeros( 10 )
                else:
                   channels_mask = np.array( [True if int(channel) in mc_channels_in_data_channel else False\
                                                for channel in data_series['fElecChannels.fChannelLocalId'][0] ] )
                   summed_wfm = np.sum( np.array( [np.array(wfm) for wfm in data_series['fElecChannels.fWFAmplitude'][0] ] )[channels_mask], axis=0 )

                summed_wfm = np.array(summed_wfm) * 9. # This scales the waveform to units of electrons. 
                summed_wfm = summed_wfm / self.analysis_config.run_parameters['Electrons/ADC [electrons]']

                if len(summed_wfm) > self.sim_wfm_length - self.sim_pretrigger_length:
                   summed_wfm = summed_wfm[0:(self.sim_wfm_length - self.sim_pretrigger_length)]

                # The blank wfm allows us to add the pre-trigger time to the simulated waveform.
                full_wfm = np.zeros(self.sim_wfm_length)
                full_wfm[ self.sim_pretrigger_length : self.sim_pretrigger_length+len(summed_wfm) ] = summed_wfm 
                if self.sim_pretrigger_length+len(summed_wfm) < self.sim_wfm_length:
                        wfm_end_length = len( full_wfm[self.sim_pretrigger_length+len(summed_wfm):] )
                        full_wfm[ self.sim_pretrigger_length+len(summed_wfm): ] = np.ones(wfm_end_length)*summed_wfm[-1]

                summed_wfm = full_wfm

                if self.add_noise and 'TileStrip' in row['ChannelType']:
                   # Downsample the noise wfm by the sampling rate ratio between sim and data    
                   downsampled_noise_wfm = noise_waveforms[ row['ChannelName'] ][::int(self.wfm_sampling_ratio)] 
                   if len(downsampled_noise_wfm) >= len(summed_wfm):
                      summed_wfm = summed_wfm + \
                                   ( downsampled_noise_wfm[0:len(summed_wfm)] * \
                                     self.analysis_config.calibration_constants['Calibration'].loc[ row['ChannelName'] ] )
                   else:
                      print('WARNING! The noise waveform (length {}) is not long enough\n'.format(len(downsampled_noise_wfm))+\
                            '         to cover the entire simulated waveform (length {}).'.format(len(summed_wfm)))
                      summed_wfm[0:len(downsampled_noise_wfm)] = summed_wfm[0:len(downsampled_noise_wfm)] +\
                                                                 downsampled_noise_wfm

                channel_waveforms.append( summed_wfm )

            return np.array( channel_ids ), np.array( channel_waveforms ), \
                   np.array( channel_types ), np.array( channel_positions )
       

        ####################################################################
        def GetFileTitle( self, filepath ):
            filename = filepath.split('/')[-1]
            filetitle = filename.split('.')[0]
            return filetitle


        ####################################################################
        def GetTotalEntries( self ):
            return self.electree.num_entries



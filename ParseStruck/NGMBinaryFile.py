############################################################################
# This file defines a class that reads the binary files produced by the 
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
import struct

class NGMBinaryFile:

     ####################################################################
     def __init__( self, input_filename=None, output_directory=None, config=None, start_stop = [None, None]):
          print('NGMBinaryFile object constructed.')

          self.start_stop = start_stop
          package_directory = os.path.dirname(os.path.abspath(__file__))
          if output_directory is None:
               self.output_directory = './'
          else:
               self.output_directory = output_directory + '/'

          if input_filename is not None:
               self.LoadBinaryFile( input_filename )
          if config is not None:
               self.channel_map = config.channel_map
          else:
               print('WARNING: No channel map file provided. Using the default one...')
               self.channel_map = pd.read_csv(package_directory + \
                                         '/channel_map_8ns_sampling.txt',skiprows=9)
          self.h5_file = None

     ####################################################################
     def LoadBinaryFile( self, filename ):
          self.filename = filename
          print('Input file: {}'.format(self.filename))


          self.run_header, self.spills_list, self.words_read, self.spill_counter = \
                                  self.ReadFile( self.filename )
          #self.run_header = run_header
          #self
          if len(self.spills_list) == 0:
               print('Spills_list is empty... no spills found in file.')     

     ####################################################################
     def GroupEventsAndWriteToHDF5( self, nevents = -1, save = True, start_stop = None ):
          
          try:
               self.spills_list
          except NameError:
               self.LoadBinaryFile( self.filename )

          if start_stop is not None:
               self.start_stop = start_stop     

          start_time = time.time()          
          file_counter = 0
          global_evt_counter = 0
          local_evt_counter = 0
          df = pd.DataFrame(columns=['Channels','Timestamp','Data','ChannelTypes','ChannelPositions'])
          output_event_list = []
          start_time = time.time()
          print('{} entries per event.'.format(len(self.channel_map)))

          spill_counter = 0
                # Note: this code assumes that the data is acquired in a way that records
                # all the active channels simultaneously. This is the typical operating mode
                # for the Stanford TPC, but may not be transferrable to other setups using the
                # Struck digitizers.
          for spill_dict in self.spills_list:

               spill_data = spill_dict['spill_data']
               num_channels = len(spill_data)
               
               # Get the number of events in the spill. The loop is there because some channels
               # are off and will have no events, which would cause problems.
               num_events = 0
               for i in range(num_channels):
                    if len(spill_data[i]['data']['events']) > num_events:
                         num_events = len(spill_data[i]['data']['events'])
               print('Bulding {} events from spill {} at {:4.4} min'.format(num_events, \
                                                                            spill_counter,\
                                                                            (time.time()-start_time)/60.))
               spill_counter += 1
               
               for i in range(num_events):
                   
                    output_dict = {'Channels': [],
                              'Timestamp': [],
                              'Data': [],
                              'ChannelTypes': [],
                              'ChannelPositions': []}
                   

                    # In the binary files, the order of channels is always sequential. Meaning, the
                    # channels go in order of (slot,chan) indexed from 0.
                    for ch, channel_data in enumerate(spill_data):

                         chan_mask = ( channel_data['card'] == self.channel_map['Board'] ) \
                                   & ( channel_data['chan'] == self.channel_map['InputChannel'] )
                         if np.sum(chan_mask) < 1:
                              print('************ ERROR IN NGMBinaryFile.py ********')
                              print('(card,chan) == ({},{}) not found in channel map'.format(\
                                        channel_data['card'], channel_data['chan'] ))
                              sys.exit(1)

                         # Strip out 'Off' channels
                         if self.channel_map['ChannelType'].loc[chan_mask].values[0] != 'Off':


                              output_dict['Channels'].append( channel_data['card']*16 + channel_data['chan'] )
                              output_dict['Timestamp'].append( \
                                   channel_data['data']['events'][i]['timestamp_full'] )
                              output_dict['Data'].append( \
                                   np.array(channel_data['data']['events'][i]['samples'], dtype=int) )
                              output_dict['ChannelTypes'].append( self.channel_map['ChannelType'].loc[chan_mask].values[0] )
                              output_dict['ChannelPositions'].append( 0. )
               
                    output_event_list.append(output_dict)
               
                    global_evt_counter += 1
                    local_evt_counter += 1
                    if local_evt_counter > 200 and save:
                         temp_df = pd.DataFrame( output_event_list[-200:] )
                         output_filename = '{}{}_{:0>3}.h5'.format( self.output_directory,\
                                                  self.GetFileTitle( self.filename ),\
                                                  file_counter )
                         temp_df.to_hdf( output_filename, key='raw' )
                         local_evt_counter = 0
                         file_counter += 1
                         print('Written to {} at {:4.4} seconds'.format(output_filename, time.time()-start_time))
               
          df = pd.DataFrame(output_event_list)
          return df 


     
     ####################################################################
     def GenerateChannelMask( self, slot_column, channel_column ):

          channel_mask = np.array(np.ones(len(slot_column),dtype=bool))
          channel_types = ['' for i in range(len(slot_column))]
          channel_positions = np.zeros(len(slot_column),dtype=int)

          for index,row in self.channel_map.iterrows():
               
               slot_mask = np.where(slot_column==row['Board'])
               chan_mask = np.where(channel_column==row['InputChannel'])
               intersection = np.intersect1d(slot_mask,chan_mask)
               if len(intersection) == 1:
                    this_index = intersection[0]
               else:
                     continue
               channel_types[this_index] = row['ChannelType']
               channel_positions[this_index] = row['ChannelPosX'] \
                                                        if row['ChannelPosX'] != 0 else row['ChannelPosY']
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
          total_entries = 0

          # Get the number of events contained in the spill
          for spill in self.spills_list:
              # Channels that are off will have 0 events. To ensure we're not
              # grabbing one of those, just get the number of events from the
              # channel with the most events.
              #print('\n\n'); print(spill); print('\n\n')
              max_evts_per_chan = 0
              for channel in spill['spill_data']:
                  #print(channel)
                  if len(channel['data']['events']) > max_evts_per_chan:
                     max_evts_per_chan = len(channel['data']['events'])

              total_entries += max_evts_per_chan
          return total_entries
    
     ####################################################################
     def ReadFile( self, filename, max_num_of_spills=-1):
         start_time = time.time()
     
         total_words_read = 0
         spills_list = []
         run_header = []
         spill_counter = 0
       
         # The basic unit in which data are stored in the NGM binary files is the
         # "spill", which consists of a full read of a memory bank on the Struck
         # board. The spill contains some header information and then all of the data stored
         # in the memory bank, sorted sequentially by card and then by channel within the card. 
         with open(filename, 'rb') as infile:
     
             # Read in the runhdr (one per file)
             for i in range(100):
                 run_header.append(hex(struct.unpack("<I", infile.read(4))[0]))
     
             first_word_of_spillhdr = hex(struct.unpack("<I", infile.read(4))[0])
     
             while True:
                 spill_time = (time.time() - start_time)  # / 60.
                 print('Reading spill {} at {:4.4} sec'.format(spill_counter, spill_time))
     
                 # this allows us to check if we're actually on a new spill or not
                 if first_word_of_spillhdr == '0xabbaabba':
                     spill_dict, words_read, last_word_read = \
                         self.ReadSpill(infile, first_word_of_spillhdr)
                 elif first_word_of_spillhdr == '0xe0f0e0f':
                     break
                 spills_list.append(spill_dict)
                 total_words_read += words_read
                 spill_counter += 1
                 first_word_of_spillhdr = last_word_read
     
                 if (spill_counter > max_num_of_spills) and \
                         (max_num_of_spills >= 0):
                     break
     
         end_time = time.time()
         print('\nTime elapsed: {:4.4} min'.format((end_time - start_time) / 60.))
     
         return run_header, spills_list, words_read, spill_counter



     ####################################################################
     def ReadSpill( self, infile, first_entry_of_spillhdr ):
         debug = False
     
         spill_dict = {}
         spill_words_read = 0
     
         spill_header = []
         spill_header.append(first_entry_of_spillhdr)
         for i in range(9):
             spill_header.append(hex(struct.unpack("<I", infile.read(4))[0]))
         spill_dict['spillhdr'] = spill_header
     
         if spill_header[0] == '0xe0f0e0f':
             return spill_dict, 0, spill_header[-1]
     
         data_list = []
         previous_card_id = 9999999
     
         while True:
             data_dict = {}
             hdrid = 0
     
             # Grab the first word, which should be either a hdrid or a phdrid
             hdrid_temp = struct.unpack("<I", infile.read(4))[0]
     
             # Break the loop if we've reached the next spill
             if hex(hdrid_temp) == '0xabbaabba' or \
                     hex(hdrid_temp) == '0xe0f0e0f':
                 last_word_read = hex(hdrid_temp)
                 break
     
             this_card_id = (0xff000000 & hdrid_temp) >> 24
             if debug:
                 print('hdrid_temp: {}'.format(hex(hdrid_temp)))
                 print('Card ID: {}'.format(this_card_id))
             data_dict['card'] = this_card_id
     
             if (this_card_id != previous_card_id) and (hdrid_temp & 0xff0000 == 0):
                 # If we've switched to a new card and are on channel 0, there should
                 # be a phdrid; two 32-bit words long.
     
                 # print('\nREADING NEXT CARD')
                 phdrid = []
                 phdrid.append(hdrid_temp)
                 phdrid.append(struct.unpack("<I", infile.read(4))[0])
                 data_dict['phdrid'] = phdrid
                 if debug:
                     print('phdrid:')
                     for val in phdrid:
                         print('\t{}'.format(hex(val)))
     
                 hdrid = struct.unpack("<I", infile.read(4))[0]
                 previous_card_id = this_card_id
             else:
                 # if not, then the hdrid_temp read in above must be the hdrid for
                 # the individual channel
                 hdrid = hdrid_temp
     
             data_dict['hdrid'] = hex(hdrid)
             if debug: print('hdrid: {}'.format(data_dict['hdrid']))
     
             channel_id = ((0xc00000 & hdrid) >> 22) * 4 + ((0x300000 & hdrid) >> 20)
             if debug: print('channelid: {}'.format(channel_id))
     
             data_dict['chan'] = channel_id
     
             channel_dict, words_read = self.ReadChannel(infile)
             data_dict['data'] = channel_dict
     
             spill_words_read += words_read
             data_list.append(data_dict)
     
         spill_dict['spill_data'] = data_list
     
         return spill_dict, spill_words_read, last_word_read





     ####################################################################
     def ReadChannel( self, infile):
          # Assumes we've already read in the hdrid
          trigger_stat_spill = []
      
          channel_dict = {}
      
          # Trigger stat. counters are defined in Chapter 4.9 of Struck manual
          # 0 - Internal trigger counter
          # 1 - Hit trigger counter
          # 2 - Dead time trigger counter
          # 3 - Pileup trigger counter
          # 4 - Veto trigger counter
          # 5 - High-Energy trigger counter
          for i in range(6):
              trigger_stat_spill.append(hex(struct.unpack("<I", infile.read(4))[0]))
          channel_dict['trigger_stat_spill'] = trigger_stat_spill
      
          # data_buffer_size stores the number of words needed to read all the
          # events for a channel in the current spill. Its size should be an integer
          # multiple of:
          # (# of header words, defined by format bits) + (# of samples/waveform)/2.
          data_buffer_size = struct.unpack("<I", infile.read(4))[0]
          channel_dict['data_buffer_size'] = data_buffer_size
          if data_buffer_size == 0:
              i = 0 
      
          total_words_read = 0 
          num_loops = 0 
          events = []
      
          while total_words_read < data_buffer_size:
              # if num_loops%10==0: print('On loop {}'.format(num_loops))
              words_read, event = self.ReadEvent(infile)
              total_words_read += words_read
              events.append(event)
              num_loops += 1
      
          channel_dict['events'] = events
      
          return channel_dict, total_words_read

       
     ####################################################################
     def ReadEvent( self, infile):
         # The "event" structure is defined in Chapter 4.6 of the Struck manual.
         # This starts with the Timestamp and ends with ADC raw data (we do not use
         # the MAW test data at this time)
     
         event = {}
         bytes_read = 0
     
         word = struct.unpack("<I", infile.read(4))[0]
         bytes_read += 4
     
         event['format_bits'] = 0xf & word
         event['channel_id'] = 0xff0 & word
         event['timestamp_47_to_32'] = 0xffff0000 & word
     
         word = struct.unpack("<I", infile.read(4))[0]
         bytes_read += 4
         event['timestamp_full'] = word | (event['timestamp_47_to_32'] << 32)
       
         # Read the various event metadata, specificed by the format bits
         if bin(event['format_bits'])[-1] == '1':
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['peakhigh_val'] = 0x0000ffff & word
             event['index_peakhigh_val'] = 0xffff0000 & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['information'] = 0xff00000 & word
             event['acc_g1'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['acc_g2'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['acc_g3'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['acc_g4'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['acc_g5'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['acc_g6'] = 0x00ffffff & word
     
         if bin(event['format_bits'] >> 1)[-1] == '1':
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['acc_g7'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['acc_g8'] = 0x00ffffff & word
     
         if bin(event['format_bits'] >> 2)[-1] == '1':
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['maw_max_val'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['maw_val_pre_trig'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['maw_val_post_trig'] = 0x00ffffff & word
     
         if bin(event['format_bits'] >> 3)[-1] == '1':
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['start_energy_val'] = 0x00ffffff & word
     
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['max_energy_val'] = 0x00ffffff & word
     
         # Read the sampling information
         word = struct.unpack("<I", infile.read(4))[0]
         bytes_read += 4
         event['num_raw_samples'] = 0x03ffffff & word
         event['maw_test_flag'] = 0x08000000 & word
         event['status_flag'] = 0x04000000 & word
     
         # Read the actual ADC samples. Note that each 32bit
         # word contains two samples, so we need to split them.
         event['samples'] = []
         for i in range(event['num_raw_samples']):
             word = struct.unpack("<I", infile.read(4))[0]
             bytes_read += 4
             event['samples'].append(word & 0x0000ffff)
             event['samples'].append((word >> 16) & 0x0000ffff)
     
         # There is an option (never used in the Gratta group to my knowledge) to have
         # the digitizers perform on-board shaping with a moving-average window (MAW)
         # and then save the resulting waveform:
         if event['maw_test_flag'] == 1:
     
             for i in range(event['num_raw_samples']):
                 word = struct.unpack("<I", infile.read(4))[0]
                 bytes_read += 4
                 event['maw_test_data'].append(word & 0x0000ffff)
                 event['maw_test_data'].append((word >> 16) & 0x0000ffff)
     
         words_read = bytes_read / 4
         return words_read, event
     

import pandas as pd
import numpy as np
import math
import csv
#import ROOT


microsecond = 1.e3
second = 1.0e9
DEBUG = True

class StruckAnalysisConfiguration:

  ###############################################################################
  def __init__( self ):
      self.channel_map = None
      self.calibration_constants = None
      self.run_parameters = None


  ###############################################################################
  def GetChannelMapFromFile( self, input_file ):
      self.channel_map = pd.read_csv( input_file, delimiter=',' )

      
  ###############################################################################
  def GetCalibrationConstantsFromFile( self, input_file ):
      self.calibration_constants = pd.read_csv( input_file, delimiter=',' )

      # Index the calibration constants by the channel name (i.e. 'X1-12')
      self.calibration_constants = self.calibration_constants.set_index('ChannelName')  


  ###############################################################################
  def GetCalibrationConstantForSoftwareChannel( self, software_channel_int ):
      mask = self.channel_map['SoftwareChannel'] == software_channel_int
      if 'TileStrip' in self.channel_map['ChannelType'].loc[mask].values[0]:
            channel_name = self.channel_map['ChannelName'].loc[mask].values[0]
            return self.calibration_constants['Calibration'].loc[channel_name]
      else:
            # I assume there's no calibration constants for SiPM channels
            return 1. 


  ###############################################################################
  def GetDecayTimeForSoftwareChannel( self, software_channel_int ):
      mask = self.channel_map['SoftwareChannel'] == software_channel_int
      if 'TileStrip' in self.channel_map['ChannelType'].loc[mask].values[0]:
            channel_name = self.channel_map['ChannelName'].loc[mask].values[0]
            return self.calibration_constants['DecayTime'].loc[channel_name]
      else:
            # I assume there's no calibration constants for SiPM channels
            return 1.e9 

  ###############################################################################
  def GetSoftwareChannelOfPMT( self ):
      mask = self.channel_map['ChannelType'] == 'PMT'
      if np.sum(mask) < 1: # If there are no PMT channels...
         return None
      else:
         return self.channel_map['SoftwareChannel'].loc[mask].values[0]


  ###############################################################################
  def GetSoftwareChannelOfPulser( self ):
      mask = self.channel_map['ChannelType'] == 'Pulser'
      if np.sum(mask) < 1: # If there are no PMT channels...
         return None
      else:
         return self.channel_map['SoftwareChannel'].loc[mask].values[0]

       
  ###############################################################################
  def GetChannelNameForSoftwareChannel( self, software_channel_int ):
      mask = self.channel_map['SoftwareChannel'] == software_channel_int
      return self.channel_map['ChannelName'].loc[mask].values[0]


  ###############################################################################
  def GetChannelTypeForSoftwareChannel( self, software_channel_int ):
      mask = self.channel_map['SoftwareChannel'] == software_channel_int
      return self.channel_map['ChannelType'].loc[mask].values[0]


  ###############################################################################
  def GetChargeChannelMask( self ):
      charge_mask = np.zeros( self.GetNumberOfChannels() )
      for index, row in self.channel_map:
          if 'TileStrip' in  row['ChannelType']:
             charge_mask[ row['SoftwareChannel'] ] = 1
      return charge_mask


  ###############################################################################
  def GetSiPMChannelMask( self ):
      sipm_mask = np.zeros( self.GetNumberOfChannels() )
      for index, row in self.channel_map:
          if 'SiPM' in  row['ChannelType']:
             sipm_mask[ row['SoftwareChannel'] ] = 1
      return sipm_mask

  ###############################################################################
  def GetDeadChannelMask( self ):
      dead_mask = np.zeros( self.GetNumberOfChannels() )
      for index, row in self.channel_map.iterrows():
          if row['IsDead']:
             dead_mask[ row['SoftwareChannel'] ] = 1
      return dead_mask


  ###############################################################################
  def GetStruckToMCChannelMap( self ):
      # This maps charge channels to channels in the MC simulation.
      # Each entry in the struck_to_mc_channel_map will be a list
      # of MC channel numbers that get mapped to that specific 
      # digitizer channel. SiPM channels will be empty lists.
      struck_to_mc_channel_map = []
      for i in range( self.GetNumberOfChannels() ):
          if 'TileStrip' in self.GetChannelTypeForSoftwareChannel(i):
             struck_to_mc_channel_map.append( self.GetMCMapForSoftwareChannel(i) )
          else:
             struck_to_mc_channel_map.append( [] )
      return struck_to_mc_channel_map 
          


  ###############################################################################
  def GetMCMapForSoftwareChannel( self, software_channel_int ):
      mask = (self.channel_map['SoftwareChannel'] == software_channel_int)
      mc_channel_mapping = str( self.channel_map['MCChannelMap'].loc[mask].values[0] )
      mc_channel_mapping_list_str = mc_channel_mapping.split(',')
      mc_channel_mapping_list_int = []

      for channel_string in mc_channel_mapping_list_str:
          try:
             mc_channel_mapping_list_int.append( int(channel_string) )
          except ValueError:
             print('Software channel {} has no associated MC channels...'.format(software_channel_int))

      return mc_channel_mapping_list_int

  ###############################################################################
  def GetRMSkeVArray( self ):
      rms_keV_array = np.zeros( self.GetNumberOfChannels() )
      for index, row in self.channel_map.iterrows():
        if 'TileStrip' in row['ChannelType']:
           channel_name = row['ChannelName']
           rms_keV_array[ row['SoftwareChannel'] ] = self.calibration_constants['RMS_keV'].loc[channel_name]
      return rms_keV_array

  ###############################################################################
  def GetRMSkeVSigmaArray( self ):
      rms_keV_sigma_array = np.zeros( self.GetNumberOfChannels() )
      for index, row in self.channel_map.iterrows():
        if 'TileStrip' in row['ChannelType']:
           channel_name = row['ChannelName']
           rms_keV_sigma_array[ row['SoftwareChannel'] ] = self.calibration_constants['RMS_keV sigma'].loc[channel_name]
      return rms_keV_sigma_array

  ###############################################################################
  def GetChannelPosXArray( self ):
      # Initialize to -1000, some unrealistically large number
      channel_pos_x_array = np.ones( self.GetNumberOfChannels() )*(-1000.) 
      for index, row in self.channel_map.iterrows():
        if 'TileStrip' in row['ChannelType']:
           channel_name = row['ChannelName']
           channel_pos_x_array[ row['SoftwareChannel'] ] = row['ChannelPosX']
      return channel_pos_x_array


  ###############################################################################
  def GetChannelPosYArray( self ):
      # Initialize to -1000, some unrealistically large number
      channel_pos_y_array = np.ones( self.GetNumberOfChannels() )*(-1000.) 
      for index, row in self.channel_map.iterrows():
        if 'TileStrip' in row['ChannelType']:
           channel_name = row['ChannelName']
           channel_pos_y_array[ row['SoftwareChannel'] ] = row['ChannelPosX']
      return channel_pos_y_array


  ###############################################################################
  def GetNumberOfChannels( self ):
      return len( self.channel_map )


  ###############################################################################
  def IsClockSourceConsistentWithSamplingRate( self, clock_source_choice ):

      NGM_sampling_freq = -1. # Initialize at unphysical value.

      if clock_source_choice == 0:
         NGM_sampling_freq = 250 # MHz
      elif clock_source_choice == 1:
         NGM_sampling_freq = 125 # MHz
      elif clock_source_choice == 2:
         NGM_sampling_freq = 62.5 # MHz
      elif clock_source_choice == 3:
         NGM_sampling_freq = 25 # MHz
      else:
         NGM_sampling_freq = -1. # Does not match any possible options

      if (NGM_sampling_freq - self.run_parameters['Sampling Rate [MHz]'])**2 > 1.:
         return False
      else:
         return True 
 

  ###############################################################################
  def GetRunParametersFromFile( self, input_file ):
      # Run parameters:
      #     Drift Length [mm]
      #     Drift Velocity [mm/us]
      #     Max Drift Time [us]
      #     Energy Start Time [us]
      #     Decay Start Time [us]
      #     Decay End Time [us]
      #     Decay Guess [us]
      #     Sampling Rate [MHz]
      #     Sampling Period [ns]
      #     Waveform Length [samples]
      #     Pretrigger Length [samples]
      #     Baseline Length [samples]
      #     Num Struck Boards

      # The input file needs two columns: 'Parameter' and 'Value'
      # We will end up with a dict called run_parameters

      temp_dataframe = pd.read_csv( input_file, delimiter=',' )
      self.run_parameters = dict(zip(temp_dataframe['Parameter'],temp_dataframe['Value']))


      # Here we actually need to calculate some stuff.
      self.run_parameters['Max Drift Time [us]'] = self.run_parameters['Drift Length [mm]'] / \
                                                   self.run_parameters['Drift Velocity [mm/us]']

      self.run_parameters['Sampling Period [ns]'] = 1.e9/(self.run_parameters['Sampling Rate [MHz]']*1.e6)


      self.run_parameters['Baseline Average Time [us]'] = self.run_parameters['Baseline Length [samples]'] * \
                                                          self.run_parameters['Sampling Period [ns]'] / 1.e3

      self.run_parameters['Energy Start Time [us]'] = (self.run_parameters['Waveform Length [samples]'] - \
                                                       self.run_parameters['Baseline Length [samples]'] ) * \
                                                      self.run_parameters['Sampling Period [ns]'] / 1.e3
                                                      # Energy is calculated using the last N samples of the waveform

      self.run_parameters['Decay Start Time [us]'] = self.run_parameters['Energy Start Time [us]']
      self.run_parameters['Decay End Time [us]'] = self.run_parameters['Decay Start Time [us]'] + \
                                                   self.run_parameters['Baseline Length [samples]']

      self.run_parameters['Decay Guess [us]'] = 200.

      # Finally, set the inversion flag
      if self.run_parameters['Sampling Rate [MHz]'] == 125:
         self.run_parameters['DoInvert'] = True
      elif self.run_parameters['Sampling Rate [MHz]'] == 62.5:
         self.run_parameters['DoInvert'] = False
      else:
         self.run_parameters['DoInvert'] = False
    
      if DEBUG:
         print('\nrun_parameters:')
         print(self.run_parameters)      


 

import os
import sys

import time
import ROOT
from ROOT import TF1
from ROOT import gROOT
from ROOT import *
#ROOT.gROOT.SetBatch(True)

from tms_utilities.digi_filter import digi_hp
from tms_utilities.digi_filter import digi_lp

import numpy as np
import pandas as pd
from scipy.fftpack import fft
from numpy import mean, sqrt, square, arange

import parse_psd_wfm
import parse_nai_wfm
import parse_tms_wfm

import subprocess
root_version = subprocess.check_output(['root-config --version'], shell=True)
isROOT6 = False
if '6.1.0' in root_version or '6.04/06' in root_version:
    print("Found ROOT 6")
    isROOT6 = True

whichDetector = 4 #raw_input('Which detector? 1 for PSD, 2 for wire array, 3 for NaI, 4 for AmBe test: ')

def print_fit_info(fit_result, fit_duration):
        
    chi2 = fit_result.Chi2()
    prob = fit_result.Prob()
    ndf = fit_result.Ndf()
    status = fit_result.Status()
    #print "fit results:"
    #print "\tchi2: %.2f" % chi2
    #print "\tn dof", ndf
    #print "\tchi2 / dof: %.2f" % (chi2/ndf)
    #print "\tprob", prob
    #print "\tstatus", status
    #print "\t%.1f seconds" % fit_duration

########################################################################
def GetChannelTypeMap():
  num_channels = 16
  channel_type_map = ['empty' for i in range(0,num_channels)]

  channel_type_map[0]  = 'Xwire'
  channel_type_map[1]  = 'Xwire'
  channel_type_map[2]  = 'Xwire'
  channel_type_map[3]  = 'Xwire'
  channel_type_map[4]  = 'Xwire'
  channel_type_map[5]  = 'Xwire'
  channel_type_map[6]  = 'Xwire'
  channel_type_map[7]  = 'Xwire'
  channel_type_map[8]  = 'Ywire'
  channel_type_map[9]  = 'Ywire'
  channel_type_map[10] = 'Ywire'
  channel_type_map[11] = 'Ywire'
  channel_type_map[12] = 'Ywire'
  channel_type_map[13] = 'Ywire'
  channel_type_map[14] = 'Ywire'
  channel_type_map[15] = 'Ywire'

  return channel_type_map
########################################################################
def GetChannelTypeMapPSD():
  num_channels = 16
  channel_type_map = ['empty' for i in range(0,num_channels)]

  channel_type_map[0] = 'PSD'

  return channel_type_map
########################################################################
def GetChannelTypeMapNaI():
  num_channels = 16 
  channel_type_map = ['empty' for i in range(0,num_channels)]

  channel_type_map[0] = 'NaI'

  return channel_type_map
########################################################################
def GetChannelTypeMapAmBeTest():
  num_channels=16
  channel_type_map = ['empty' for i in range(0,num_channels)]
  channel_type_map[1] = 'PSD'
  channel_type_map[0] = 'NaI'
  return channel_type_map

def GetChannelTypeMapAmBeCoincidence():
  num_channels=32
  channel_type_map = ['empty' for i in range(0,num_channels)]
#  channel_type_map[1]  = 'Xwire'
#  channel_type_map[0]  = 'Xwire'
#  channel_type_map[3]  = 'Xwire'
#  channel_type_map[2]  = 'Xwire'
#  channel_type_map[5]  = 'Xwire'
#  channel_type_map[4]  = 'Xwire'
#  channel_type_map[7]  = 'Xwire'
#  channel_type_map[6]  = 'Xwire'
#  channel_type_map[9]  = 'Ywire'
#  channel_type_map[8]  = 'Ywire'
#  channel_type_map[11] = 'Ywire'
#  channel_type_map[10] = 'Ywire'
#  channel_type_map[13] = 'Ywire'
#  channel_type_map[12] = 'Ywire'
#  channel_type_map[15] = 'Ywire'
#  channel_type_map[14] = 'Ywire'
  channel_type_map[17] = 'PSD'
  channel_type_map[16] = 'NaI'
  return channel_type_map

def GetChannelTypeMapAmBeCoincidenceBigNaI():
  num_channels=16
  channel_type_map = ['empty' for i in range(0,num_channels)]
  channel_type_map[0] = 'NaI'
  channel_type_map[1] = 'NaI'
  channel_type_map[3] = 'PSD'
  return channel_type_map


########################################################################
def ProcessFile( datadir, thisfile, num_events = -1, save_waveforms = False, event_list = None):

    filename = datadir + thisfile

    outfilename = filename.split('.')[0] + '_.pkl'
    print(outfilename)

    file_id = thisfile.split('/')[1].split('_')[2]

    #print "processing file: ", filename
    #if whichDetector == '1':
    #  channel_type_map = GetChannelTypeMapPSD()
    #if whichDetector == '2':
    #  channel_type_map = GetChannelTypeMap()
    #if whichDetector == '3':
    #  channel_type_map = GetChannelTypeMapNaI()
    #if whichDetector == '4':
    channel_type_map = GetChannelTypeMapAmBeCoincidence()
    #channel_type_map = GetChannelTypeMapAmBeCoincidenceBigNaI()

    #channel_type_map = GetChannelTypeMap()
    num_channels = len(channel_type_map)
    #print('Num channels: {}'.format(num_channels))

    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    # open the root file and grab the tree
    root_file = ROOT.TFile(filename)
    tree = root_file.Get("HitTree")
    n_entries = tree.GetEntries()
    #print('{} entries in HitTree'.format(n_entries))


    # Figure out how many records to process to get the right
    # number of events
    num_records_to_process = num_events * num_channels
    if num_records_to_process > n_entries or num_events < 0:
       num_records_to_process = n_entries
       num_events = n_entries/num_channels

    output_dataframe = pd.DataFrame()

    i_entry = 0
    i_event = 0
    events_finished = 0
    print('File ID: {}'.format(file_id))

    print('Number of events to process: {}'.format(num_events))
    while events_finished < num_events:
    #for i_entry in xrange(n_entries):
        if i_event % 500 == 0:
          print('Processing event {}'.format(i_event))
           
        output_series = pd.Series()
        output_series['File ID'] = np.int64(file_id)
        output_series['Event ID'] = np.float64(i_event)
        #if i_event == 87:
         #print('{}, {}'.format(output_series['File ID'],output_series['Event ID']))
         #print('output_series[fileid] is type {}'.format( str(type(output_series['File ID'])) ))
         #print('output_series[eventid] is type {}'.format( str(type(output_series['Event ID'])) ))
         #print('event_list[fileid] is type {}'.format(str(type(event_list['File ID'].iloc[0] ))))
         #print('event_list[eventid] is type {}'.format(str(type(event_list['Event ID'].iloc[0]))))


         #if np.int64( output_series['File ID'] ) in set( event_list['File ID'] ): 
         #   print('Found file ID!')
         #if np.float64(output_series['Event ID']) in set( event_list[event_list['File ID'] == np.int64(output_series['File ID'])]['Event ID'] ):
         #   print('Found Event ID!')

        this_event_timestamp = 0
        is_good_event = True


 
        for i_channel in xrange(num_channels):

            #print('output_series[File ID]')
            #print(output_series['File ID'])
            #print('output_series[Event ID]')
            #print(output_series['Event ID'])
            #print( set(event_list['File ID']) )

            if event_list is not None and\
               ( output_series['File ID'] not in set( event_list['File ID'] ) or\
                 output_series['Event ID'] not in set( event_list[ event_list['File ID']==output_series['File ID'] ]['Event ID'] ) ):

                  i_entry = i_entry + 1
                  
                  if i_channel == 0:
                     is_good_event = False
                  continue
                  
            else:
               if i_channel == 0 and event_list is not None:
                 print('Event {} found in event list. Processing...'.format(i_event))

               tree.GetEntry(i_entry)
   
               slot_num = tree.HitTree.GetSlot()
               ch_num =  tree.HitTree.GetChannel() + slot_num*16
               gate_size = tree.HitTree.GetGateCount()
   
               if i_channel == 0: this_event_timestamp = tree.HitTree.GetRawClock()
               if tree.HitTree.GetRawClock() != this_event_timestamp:
                  is_good_event = False
                  print('Incomplete event! Entry: {}'.format(i_entry))
                  break
               else:
                  is_good_event = True
   
               #print('entry {}, channel {}'.format(i_entry, ch_num))
   
               # Channel 6 is noisy, so we ignore it for now. 
               # (May have been destroyed by a discharge or something)
               if ch_num==6:
                   i_entry = i_entry+1
                   continue
   
               wfm_length = tree.HitTree.GetNSamples()
    
               graph = tree.HitTree.GetGraph()
               channel_wfm = np.array([graph.GetY()[isamp] for isamp in xrange(graph.GetN())])
   
               #print('Processing channel: {}'.format(ch_num))
               if channel_type_map[ch_num] == 'PSD':
                  psd_reduced_data = parse_psd_wfm.ParsePSDWfm( channel_wfm, save_waveform=save_waveforms )
                  for colname in psd_reduced_data.index:
                     new_colname = 'PSD ' + colname
                     output_series[new_colname] = psd_reduced_data[colname]
   
               if channel_type_map[ch_num] == 'Xwire':
                  xwire_reduced_data = parse_tms_wfm.ParseTMSXwireWfm( channel_wfm, save_waveform = save_waveforms )
                  for colname in xwire_reduced_data.index:
                    new_colname = 'X{} '.format(ch_num) + colname
                    output_series[new_colname] = xwire_reduced_data[colname]
   
               if channel_type_map[ch_num] == 'Ywire':
                  ywire_reduced_data = parse_tms_wfm.ParseTMSYwireWfm( channel_wfm, save_waveform = save_waveforms )
                  for colname in ywire_reduced_data.index:
                    new_colname = 'Y{} '.format(ch_num) + colname
                    output_series[new_colname] = ywire_reduced_data[colname]
               
               if channel_type_map[ch_num] == 'NaI':
                  nai_reduced_data = parse_nai_wfm.ParseNaIWfm( channel_wfm, save_waveform = save_waveforms )
                  for colname in nai_reduced_data.index:
                    new_colname = 'NaI {} '.format(ch_num) + colname
                    output_series[new_colname] = nai_reduced_data[colname]
   
               else:
                  # For now, do nothing here.
                  i_entry += 1
                  continue
   
               #print(psd_reduced_data)
   
   
               i_entry += 1
        if is_good_event:
          output_dataframe = output_dataframe.append( output_series, ignore_index=True )
          events_finished = events_finished + 1 
        i_event += 1
        if i_entry >= n_entries:
           print('Breaking...')
           break
 
    #basename = os.path.basename(filename)
    #basename = os.path.splitext(basename)[0]
    #basename = basename.split("_")[2]
    #outfilename = "./outfiles/out_new_5kV_%s.root" % basename
    #outfilename = "./outfiles/out_new_%s.root" % basename

    #outfile=ROOT.TFile(outfilename,"RECREATE")

    output_dataframe.to_pickle(outfilename)
    return output_dataframe


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('arguments: [sis root files]')
        sys.exit(1)

    for filename in sys.argv[1:]:
        process_file(filename)




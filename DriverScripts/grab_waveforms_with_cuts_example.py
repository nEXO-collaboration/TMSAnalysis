import numpy as np
from  matplotlib import pyplot as plt
import pandas as pd
import time

import histlite as hl
import sys
import os
import pickle

sys.path.append('/g/g20/lenardo1/software/')
from TMSAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration
from TMSAnalysis.WaveformAnalysis import Waveform

#########################################################################################
if len(sys.argv) != 3:
   print('\nERROR: incorrect number of arguments.\n')
   print('Usage:')
   #print('\tpython get_waveforms_with_cuts.py <event_start> <output_dir>\n\n')
   print('\tpython get_waveforms_with_cuts.py <file_idx> <output_dir>\n\n')
   sys.exit()

evt_start = int(sys.argv[1])
file_idx = int(sys.argv[1])
output_dir = sys.argv[2]
#########################################################################################

channel_map_file = '/g/g20/lenardo1/software/TMSAnalysis/config/Channel_Maps_Run30_RnPoAlphaEffTest.csv'
run_parameters_file = '/g/g20/lenardo1/software/TMSAnalysis/config/Run_Parameters_Run30_RnPoTest.csv'
calibrations_file = '/g/g20/lenardo1/software/TMSAnalysis/config/Calibrations_Xe_Run11b.csv'

analysis_config = StruckAnalysisConfiguration.StruckAnalysisConfiguration()
analysis_config.GetChannelMapFromFile('/g/g20/lenardo1/software/TMSAnalysis/config/Channel_Maps_Run30_RnPoAlphaEffTest.csv',sheet='')
analysis_config.GetRunParametersFromFile('/g/g20/lenardo1/software/TMSAnalysis/config/Run_Parameters_Run30_RnPoTest.csv',sheet='')
analysis_config.GetCalibrationConstantsFromFile('/g/g20/lenardo1/software/TMSAnalysis/config/Calibrations_Xe_Run11b.csv')

channel_names = [chname for chname in analysis_config.channel_map['ChannelName'] if 'X' in chname or 'Y' in chname]
print('{} channels being grabbed:'.format(len(channel_names)))

rundir = '/p/lustre1/jacopod/30th/'

datadir = rundir + '20200923_Afternoon_AfterSixthInjection/analysis_500ns_new_calib/'

#infilename = rundir + '20200923_Afternoon_AfterSixthInjection/analysis_500ns_new_calib/reduced_added.p'
infiles = [filename for filename in os.listdir(datadir) if filename.endswith('.h5')]

start_time = time.time()


infilename = datadir + infiles[file_idx]
print('Loading HDF5 file...')
#df = pd.read_pickle(infilename)
df = pd.read_hdf(infilename)
print('----> {} loaded ({} events).\n'.format(infiles[file_idx],len(df)))

mask = (df['TotalTileEnergy'] > 2.e3)&(df['TotalTileEnergy'] < 3e3)
cut_index = df.loc[mask].index


waveforms_dict = {}
for ch in channel_names:
    waveforms_dict[ch] = []

# Cut loop
for i in range(evt_start,evt_start+200):
    if len(df.loc[mask]) <= i:
       break
    print('Getting event {}'.format(cut_index[i]))
    
    event = Waveform.Event(infilename,\
                       rundir + '20200923_Afternoon_AfterSixthInjection/raw_data/',cut_index[i],\
                       run_parameters_file,\
                       calibrations_file,\
                       channel_map_file)
    
    raw_evt = df.loc[cut_index[i]]
    
    for ch in channel_names:
        colname = 'TileStrip {} Charge Energy'.format(ch)
        
        if raw_evt[colname] > 500.:
            waveforms_dict[ch].append(event.waveform[ch].data)
    
    print('----> Done with event {} at {:4.4} min\n'.format(\
                                cut_index[i],\
                                (time.time()-start_time)/60.))

# Save output to pkl file
with open('{}/waveforms_dict_{}.pkl'.format(output_dir,evt_start),'wb') as pklfile:
    pickle.dump(waveforms_dict,pklfile)

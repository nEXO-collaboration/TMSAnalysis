###########################################################################
# This script is for submitting processing jobs to the Farmshare cluster. 
###########################################################################

import tmsanalysis as tms 
import pandas as pd
import time
import sys

datadir = '/farmshare/user_data/blenardo/struck_data/'
#filename = 'tier1_SIS3316Raw_20180925171528_Cs137_PSDScintTest_TriggThroughDisc_62_5MHz_1-ngm.root'
datadir = datadir + sys.argv[1] + '/'

filename = sys.argv[2] #+ '_{:04d}.root'.format( int(sys.argv[2]) )
print('================================================================')
print('Opening file...')
print(filename)

neutron_peak_events = pd.read_csv('/home/blenardo/neutron_data_10-10-2018_event_lists/neutron_peak_event_list.csv')

start_time = time.time()
dfout = tms.ProcessFile(datadir,filename,save_waveforms=True,num_events=-1,event_list=neutron_peak_events)
end_time = time.time()

print('Time elapsed: {}s ({} min)'.format(end_time-start_time,(end_time-start_time)/60.))

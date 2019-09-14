import tmsanalysis as tms 
import pandas as pd
import numpy as np
import time

datadir = '/farmshare/user_data/blenardo/struck_data/neutron_overnight_10-10-2018/'
#filename = 'tier1_SIS3316Raw_20180925171528_Cs137_PSDScintTest_TriggThroughDisc_62_5MHz_1-ngm.root'
filename = 'tier1_SIS3316Raw_20181010054809_Ambe_slot2_CH1NaI1700V_CH0PSD1100V_slot1_TMS_PSDTrigg150mVTh_PSDX10_125MHz_TMSCa8kV_30in_10_1_8in_1-ngm.root'
datafile = datadir + filename

neutron_peak_events = pd.read_csv('../neutron_data_10-10-2018_event_lists/neutron_peak_event_list.csv')


start_time = time.time()
dfout = tms.ProcessFile( datadir, filename, save_waveforms=True, num_events=-1, event_list = neutron_peak_events )
end_time = time.time()

print('Time elapsed: {}s ({} min)'.format(end_time-start_time,(end_time-start_time)/60.))

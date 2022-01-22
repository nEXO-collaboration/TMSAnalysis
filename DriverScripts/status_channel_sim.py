#########################################################################################
#											#
#       This script produces a pickled dict containing the status of the channels	#
#       (mean and RMS of the baseline) for the simulation to be compared		#
#	with a particular dataset. The first argument parsed is the data reduced	#
#	file from where the status of the channels is pulled, while the second is the	#
#	folder where the pickled file will be saved.					#
#       In general the location for the pickled file is where the simulated raw data	#
#       are located (sim tier1 folder).							#
#											#
#					Usage:						#
#    python status_channel_sim.py path_to_reduced /path/to/input/sim_tier1_directory/	#
#											#
#########################################################################################



import pandas as pd
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('data_df', type=str, help='data df from which pull the channel status')
parser.add_argument('sim_raw_folder', type=str, help='folder to store the file')
args = parser.parse_args()
this_file  = args.data_df
output_dir = args.sim_raw_folder
ch_list = {}


df = pd.read_hdf(this_file)
for k in df.keys():
	if ('TileStrip' in k) and ('Charge Energy' in k):
		if np.mean(df[k])<5:
		#I arbitrarily chose 5 ADC counts as a threshold for the active channel
			ch_list[k[10:-14]] = [np.mean(df['%s Baseline'%k[:-14]]),np.mean(df['%s Baseline RMS'%k[:-14]])]

with open(output_dir + 'channel_status.p','wb') as f:
	pickle.dump(ch_list,f)

#########################################################################################
#
#	This script fetches all the reduced .h5 in the folder, producing
#	one dataframe for the run. I noticed that it's faster than loading
#	dynamically each file (in case of the entire run is needed)
#
#					Jacopo
#
#					Usage:
#			python add_into_one_df.py path_to_reduced
#
#########################################################################################


import glob, sys, time, os
import pandas as pd



start_time = time.time()
df_list = []
path_to_folder = sys.argv[1]

if path_to_folder[-1] != '/':
	path_to_folder += '/'


for i,fname in enumerate(sorted(glob.glob(path_to_folder + '*reduced.h5'))):
	if i%100 == 0:
		print('{} files appended in {:.1f}'.format(i,(time.time() - start_time)))
	df_list.append(pd.read_hdf(fname))
df = pd.concat(df_list,ignore_index=True)
df.to_pickle( path_to_folder + 'reduced_added.p')
#df.to_hdf( path_to_folder + 'reduced_added.h5',key='k')

full_folder = os.path.dirname(os.path.abspath(path_to_folder))
os.system('chmod -R 775 {}'.format(full_folder))

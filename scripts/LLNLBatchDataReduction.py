#!/usr/bin/python3

#########################################################################################
#											#
#	This script submits job to LLNL batch system, once an in input			#
#	folder is passed it scans for all the tier1 file there. After the		#
#	uproot venv is activated, one job per tier1 file is submitted.			#
#	The job option can be found and modified in the variable cmd_options.		#
#											#
#					Usage:						#
#	python LLNLBatchDataReduction.py path_to_tier1 path_to_reduced path_to_config	#
#											#
#########################################################################################



import os, sys, time, glob

path_to_tier1 	= sys.argv[1]
path_to_reduced = sys.argv[2]
path_to_config 	= sys.argv[3]


if not os.path.isdir(path_to_tier1):
   print(path_to_tier1)
   print('Not a directory! Skipping...')
   exit()

flist = glob.glob('{}tier1*.root'.format(path_to_tier1))

for i,fname in enumerate(flist):
	fname_stripped = (fname.split('/')[-1]).split('.')[0]
	outfile = '{}{}_reduced.h5'.format(path_to_reduced,fname_stripped)
	if os.path.exists(outfile):
		print('file {}_reduced.h5 already exists'.format(fname_stripped))
		continue
	activate_venv = 'source $HOME/uproot/bin/activate && source $HOME/software/TMSAnalysis/setup.sh'
	cmd_options = '--export=ALL -p pbatch  -t 02:00:00 -n 1 -J {} -o {}{}.out'.format(i,path_to_reduced,fname_stripped)
	exe = 'python $HOME/software/TMSAnalysis/scripts/reduce_data.py {} {} {}'.format(fname,path_to_reduced,path_to_config)
	cmd_full = '{} && sbatch {} --wrap=\'{}\''.format(activate_venv,cmd_options,exe)
	os.system(cmd_full)
	print('job {} sumbitted'.format(i))

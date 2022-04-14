#!/usr/bin/python3

#################################################################################################
#												#
#	This script submits job to LLNL batch system, once an in input				#
#	folder is passed it scans for all the tier1 file there. After the			#
#	uproot venv is activated, one job per tier1 file is submitted.				#
#	The job option can be found and modified in the variable cmd_options.			#
#												#
#					Usage:							#
#	python LLNLBatchDataReduction.py path_to_tier1 path_to_reduced path_to_config (--sim)	#
#												#
#################################################################################################



import argparse, os, sys, time, glob



parser = argparse.ArgumentParser()
parser.add_argument('inputdir', type=str, help='path of input files')
parser.add_argument('outputdir', type=str, help='location to put output files')
parser.add_argument('configdir', type=str, help='location where config files are stored. '+\
                                                'Note that this must include three config files:\n'+\
                                                ' Run_Parameters,\n'+\
                                                ' Calibrations, and\n'+\
                                                ' Channel_Map')
parser.add_argument('--sim', help='Simulation flag', action='store_true')
parser.add_argument('--noise', help='Simulation flag', action='store_true')
args = parser.parse_args()
path_to_tier1   = args.inputdir
path_to_reduced = args.outputdir
path_to_config = args.configdir

if not os.path.isdir(path_to_tier1):
   print(path_to_tier1)
   print('Not a directory! Skipping...')
   exit()

if path_to_tier1[-1] != '/':
	path_to_tier1 += '/'

if path_to_reduced[-1] != '/':
	path_to_reduced += '/'

if not os.path.exists(path_to_reduced):
	print('No output directory found - Creating a new one')
	os.makedirs(path_to_reduced)


flist = glob.glob('{}*.root'.format(path_to_tier1))

for i,fname in enumerate(flist):
	fname_stripped = (fname.split('/')[-1]).split('.')[0]
	outfile = '{}{}_reduced.h5'.format(path_to_reduced,fname_stripped)
	if os.path.exists(outfile):
		print('file {}_reduced.h5 already exists'.format(fname_stripped))
		continue
	activate_venv = 'source $HOME/uproot/bin/activate && source $HOME/software/StanfordTPCAnalysis/setup.sh'
	cmd_options = '--export=ALL -p pbatch  -t 5-10:00:00 -n 1 -J {} -o {}{}.out'.format(i,path_to_reduced,fname_stripped)
	exe = 'python $HOME/software/StanfordTPCAnalysis/DriverScripts/reduce_data.py {} {} {}'.format(fname,path_to_reduced,path_to_config)
	if args.sim:
		exe += ' --sim'
		if args.noise:
			exe += ' --noise'
	cmd_full = '{} && sbatch {} --wrap=\'{}\''.format(activate_venv,cmd_options,exe)
	os.system(cmd_full)
	print('job {} sumbitted'.format(i))

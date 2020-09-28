#########################################################################################
#
#       This script converts to ROOT files the binary files from the struck in batch.
#       In order to run this script the NGM software needs to be installed and 
#       (I think) the the TMS virtual environment deactivated
#
#                                       Jacopo
#
#                                       Usage:
#                       python LLNLBatchRootify.py binary_file_dir output_dir
#
#########################################################################################



import argparse, os, sys, time, glob



parser = argparse.ArgumentParser()
parser.add_argument('inputfolder', type=str, help='binary files location')
parser.add_argument('outputdir', type=str, help='location to put output files')
args = parser.parse_args()
path_to_bin   = args.inputfolder
path_to_tier1 = args.outputdir

if not os.path.isdir(path_to_bin):
   print(path_to_bin)
   print('Not a directory! Skipping...')
   exit()

if path_to_bin[-1] != '/':
	path_to_bin += '/'

if path_to_tier1[-1] != '/':
	path_to_tier1 += '/'

if not os.path.exists(path_to_tier1):
	print('No output directory found - Creating a new one')
	os.makedirs(path_to_tier1)


flist = glob.glob('{}*1.bin'.format(path_to_bin))

for i,fname in enumerate(flist):
	fname_stripped = (fname.split('/')[-1]).split('.')[0]
	outfile = 'tier1_' + fname_stripped + '-ngm.root'
	if os.path.exists(path_to_tier1 + outfile):
		print('file {} already exists'.format(outfile))
		continue
	cmd_options = '--export=ALL -p pbatch  -t 02:00:00 -n 1 -J {} -o {}{}.out'.format(i,path_to_tier1,fname_stripped)
	exe = 'python $HOME/software/TMSAnalysis/DriverScripts/toRoot.py {} {}'.format(path_to_tier1,fname)
	cmd_full = 'sbatch {} --wrap=\'{}\''.format(cmd_options,exe)
	os.system(cmd_full)
	print('job {} sumbitted'.format(i))

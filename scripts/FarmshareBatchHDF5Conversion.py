#!/usr/bin/python3

import os
import sys
import time

print(sys.argv[1])
datapath = '/farmshare/user_data/blenardo/struck_data/29th_LXe/'
datapath = datapath + sys.argv[1] + '/'

if not os.path.isdir(datapath):
   print('Not a directory! Skipping...')
   exit()

files = os.listdir(datapath)
rootfiles = [f for f in files if f.endswith('.root')]

num_files = len(rootfiles)

print('{} files found:'.format(num_files))
#for thisfile in rootfiles:
#	print(thisfile)


outdir = datapath + '/{}/Out'.format('Processing')
subdir = datapath + '/{}/Sub'.format('Processing')
procdir = datapath + '/Processing/'

try: 
  os.mkdir(procdir)
except:
  print('{} already exists.'.format('Processing/'))
#  exit()

try:
  os.mkdir(outdir)
except Exception as e:
  print(e)
  print('Output directory exists')

try:
  os.mkdir(subdir)
except Exception as e:
  print(e)
  print('Submission directory exists')


for num in range(0,num_files):

        thisfile_base = rootfiles[num].split('.')[0]
        queue_list = os.popen('squeue -u blenardo').read()
        num_jobs_in_queue = queue_list.count('\n') - 1
        #if num_jobs_in_queue > 20:
        #   print('Num jobs is over 20, sleeping for 30s...')
        #   time.sleep(30)
        #   continue

        print(sys.argv[1] + '\t' + rootfiles[num])

        outfilename = '{}/{}_{:04d}.out'.format(outdir,sys.argv[1],num)

        scriptfilename = '{}/{}_{:04d}.sub'.format(subdir,sys.argv[1],num)
        thescript = "#!/bin/bash\n" + \
		"#SBATCH -t 02:00:00\n" + \
		"#SBATCH -e " + outfilename + "\n" + \
		"#SBATCH -o " + outfilename + "\n" + \
		"#SBATCH --export=ALL \n" + \
		"export STARTTIME=`date +%s`\n" + \
		"echo Start time $STARTTIME\n" + \
		"source activate myenv\n" + \
		"python ./TMSAnalysis/convert_data_to_hdf5.py "  + datapath + rootfiles[num] + ' ' + datapath + '\n' + \
		"export STOPTIME=`date +%s`\n" + \
		"echo Stop time $STOPTIME\n" + \
		"export DT=`expr $STOPTIME - $STARTTIME`\n" + \
		"echo CPU time: $DT seconds\n"
	
        scriptfile = open( scriptfilename, 'w' )
        scriptfile.write( thescript )
        scriptfile.close()
        os.system( "sbatch " + scriptfilename )

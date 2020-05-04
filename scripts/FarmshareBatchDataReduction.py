#!/usr/bin/python3
import os
import sys
import time
import glob


script_dir = '/g/g20/lenardo1/software/TMSAnalysis/scripts/'
config_dir = '/g/g20/lenardo1/software/TMSAnalysis/config/'

#print(sys.argv[1])
datapath = '/farmshare/user_data/blenardo/struck_data/29th_LXe/'
datapath = '/p/lustre1/lenardo1/stanford_teststand/29th_LXe/'
print(datapath)
datapath = datapath + sys.argv[1] + '/'
print(datapath)

if not os.path.isdir(datapath):
   print(datapath)
   #print('Butt Not a directory! Skipping...')
   exit()

files = os.listdir(datapath)
rootfiles = [f for f in files if f.endswith('.root')]

num_files = len(rootfiles)

print('{} files found:'.format(num_files))
#for thisfile in rootfiles:
#	print(thisfile)


reduceddir = datapath + '/Reduced_April162020/'
outdir = reduceddir + 'Out/'
subdir = reduceddir + 'Sub/'
procdir = datapath + '/Processing/'

#try: 
#  os.mkdir(procdir)
#except:
#  print('{} already exists.'.format('Processing/'))
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

try: 
  os.mkdir(reduceddir)
except:
  print('Reduced dir already exists.')
#  exit()

for num in range(0,num_files):
#for num in range(0,1):
        #if rootfiles[num].endswith('_reduced.h5'):
        #      continue

        rootfile_base = rootfiles[num].split('.')[0]

        h5files = [f for f in glob.glob(procdir + rootfile_base + '*') if f.endswith('.h5')]

        #queue_list = os.popen('squeue -u blenardo').read()
        #num_jobs_in_queue = queue_list.count('\n') - 1
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
		"for h5file in $(ls " + procdir + rootfile_base + '*' + ')\n' + \
		"do python " + script_dir + "reduce_data.py $h5file " + reduceddir + ' ' + config_dir + '\n' + \
		"done\n" + \
		"export STOPTIME=`date +%s`\n" + \
		"echo Stop time $STOPTIME\n" + \
		"export DT=`expr $STOPTIME - $STARTTIME`\n" + \
		"echo CPU time: $DT seconds\n"
	
        scriptfile = open( scriptfilename, 'w' )
        scriptfile.write( thescript )
        scriptfile.close()
        os.system( "sbatch " + scriptfilename )

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
	cmd_options = '--export=ALL -p pbatch  -t 02:00:00 -n 1 -J {} -o {}{}.out'.format(i,path_to_reduced,i)
	exe = 'python $HOME/software/TMSAnalysis/scripts/reduce_data.py {} {} {}'.format(fname,path_to_reduced,path_to_config)
	cmd_full = '{} && sbatch {} --wrap=\'{}\''.format(activate_venv,cmd_options,exe)
	os.system(cmd_full)
	print('job {} sumbitted'.format(i))
	

exit()


'''


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
		"for h5file in $(ls " + datapath + rootfile_base + '*' + ')\n' + \
		"do python reduce_data.py " + '$h5file ' + datapath + '\n' + \
		"done\n" + \
		"export STOPTIME=`date +%s`\n" + \
		"echo Stop time $STOPTIME\n" + \
		"export DT=`expr $STOPTIME - $STARTTIME`\n" + \
		"echo CPU time: $DT seconds\n"
	
        scriptfile = open( scriptfilename, 'w' )
        scriptfile.write( thescript )
        scriptfile.close()
        os.system( "sbatch " + scriptfilename )'''

#########################################################################################
#											#
#	This script checks for errors in out file from the reduce batch job.		#
#	It generates a log where in each line it is reported the filename		#
#	and the error									#
#					Usage:						#
#			python check_batch_output.py path_to_reduced			#
#											#
#########################################################################################



import sys,glob

path_to_file = sys.argv[1]


def get_line(f_object):
	for l in f_object.readlines():
		if 'file' in l:
			return l.split('/')[-1][:-1]


name_array = []
out_log = []
for out_fname in glob.glob('{}tier1*.out'.format(path_to_file)):
	with open(out_fname,'r') as f:
		try:
			line = f.readlines()[-1]
			if 'Error' in line:
				with open(out_fname,'r') as f_object:
					fname = get_line(f_object)
					name_array.append('{}: {}'.format(fname,line))

		except IndexError:
			continue

with open('{}batch_job.log'.format(path_to_file),'w') as f_out:
	if name_array == []:
		f_out.write('No errors in this folder\n')
	else:
		for title in sorted(name_array):
			f_out.write('{}'.format(title))

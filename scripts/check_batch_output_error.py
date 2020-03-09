import sys,glob

path_to_file = sys.argv[1]


def get_file_name(f_object):
	for l in f_object.readlines():
		if 'file' in l:
			return l.split('/')[-1][:-1]


name_array = []
for out_fname in glob.glob('{}[0-9]*.out'.format(path_to_file)):
	with open(out_fname,'r') as f:
		line = f.readlines()[-1]
		if 'Error' in line:
			with open(out_fname,'r') as f_object:
				fname = get_file_name(f_object)
				name_array.append(fname)

print(sorted(name_array))

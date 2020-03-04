from TMSAnalysis.ParseStruck import NGMRootFile
import sys



###############################################################################
# Check arguments and load inputs
if len(sys.argv) == 3:
	input_file = sys.argv[1]
	output_dir = sys.argv[2]
        if not os.path.exists(input_file):
                sys.exit('\nERROR: input file does not exist\n')
        if not os.path.exists(output_dir):
                sys.exit('\nERROR: path to output_dir does not exist\n')
else:
        print('\n\nERROR: convert_data_to_hdf5.py requires 2 arguments\n')
        print('Usage:')
        print('\tpython convert_data_to_hdf5.py <input_file> </path/to/output/directory/>')
        sys.exit('\n')


##############################################################################
# Load the NGM file and do the conversion.
print('Input file: {}'.format(input_file))
print('Output directory {}'.format(output_dir))

file = NGMRootFile.NGMRootFile( input_filename = input_file,\
				output_directory = output_dir,\
				channel_map_file = 'channel_map_8ns_sampling_LONGTPC2.txt')
file.GroupEventsAndWriteToHDF5()

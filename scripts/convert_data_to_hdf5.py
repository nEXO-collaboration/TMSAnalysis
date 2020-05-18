from TMSAnalysis.ParseStruck import NGMRootFile
from TMSAnalysis.ParseSimulation import NEXOOfflineFile
from TMSAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration

import sys
import os


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

path_components = sys.argv[0].split('/')
true_path = ''
for component in path_components[:-1]:
	true_path += component + '/'

IS_SIMULATION = True

if IS_SIMULATION:
	config_dir = '/g/g20/lenardo1/software/TMSAnalysis/config/'
	analysis_config = StruckAnalysisConfiguration.StruckAnalysisConfiguration()
	analysis_config.GetRunParametersFromFile( config_dir +  'Run_Parameters_Xe_Run29_SimCompatible.csv' )
	analysis_config.GetCalibrationConstantsFromFile( config_dir + 'Calibrations_Xe_Run11b.csv' )
	analysis_config.GetChannelMapFromFile( config_dir + 'Channel_Map_Xe_Run29_MCIncluded.csv' )
	infile = NEXOOfflineFile.NEXOOfflineFile( input_filename = input_file,\
						output_directory = output_dir,\
						config = analysis_config,\
						add_noise=False)
else:
	infile = NGMRootFile.NGMRootFile( input_filename = input_file,\
				output_directory = output_dir,\
				channel_map_file = true_path + '/channel_map_8ns_sampling_LONGTPC2.txt')
print('Channel map loaded:')
print(infile.channel_map)
print('\n{} active channels.'.format(len(infile.channel_map)))

infile.GroupEventsAndWriteToHDF5()

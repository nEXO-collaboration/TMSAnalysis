from TMSAnalysis.DataReduction import DataReduction
import sys
import os

###############################################################################
# Check arguments and load inputs
if len(sys.argv) == 4:
	this_file = sys.argv[1]
	output_dir = sys.argv[2]
	config_dir = sys.argv[3]
	if not os.path.exists(this_file):
		sys.exit('\nERROR: input file does not exist\n')
	if not os.path.exists(output_dir):
		sys.exit('\nERROR: path to output_dir does not exist\n')
else:
	print('\n\nERROR: reduce_data.py requires 3 arguments\n')
	print('Usage:')
	print('\tpython reduce_data.py <input_file> </path/to/output/directory/> </path/to/configuration/files/>')
	sys.exit('\n')


###############################################################################
print('\nReducing {}'.format( this_file ))

DataReduction.ReduceH5File( this_file, output_dir,\
				config_dir + '/Run_Parameters_Xe_Run29.csv',\
				config_dir + '/Calibrations_Xe_Run11b.csv',\
				config_dir + '/Channel_Map_Xe_Run29.csv',\
				input_baseline=-1,input_baseline_rms=-1,\
				fixed_trigger=True,fit_pulse_flag=False)

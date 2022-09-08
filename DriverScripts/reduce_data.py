from StanfordTPCAnalysis.DataReduction import DataReduction
import argparse
import sys
import os

###############################################################################
# Check arguments and load inputs
parser = argparse.ArgumentParser()
parser.add_argument('inputfile', type=str, help='path + name of input file')
parser.add_argument('outputdir', type=str, help='location to put output files')
parser.add_argument('configdir', type=str, help='location where config files are stored. '+\
						'Note that this must include three config files:\n'+\
						' Run_Parameters,\n'+\
						' Calibrations, and\n'+\
						' Channel_Map')
parser.add_argument('--sim', help='Simulation flag', action='store_true')
parser.add_argument('--save_raw', help='Save HDF5 of raw data', action='store_true',default=False)
args = parser.parse_args()
this_file  = args.inputfile
output_dir = args.outputdir
config_dir = args.configdir
save_hdf5 = args.save_raw
input_foldername = os.path.dirname(this_file)
print('\n\nInput foldername:')
print(input_foldername)
channel_status_file = input_foldername + '/channel_status.p'

#if all((this_file,output_dir,config_dir)):
if output_dir[-1] != '/':
	output_dir += '/'

if not os.path.exists(this_file):
	sys.exit('\nERROR: input file does not exist\n')

if not os.path.exists(output_dir):
	print('No output directory found - Creating a new one')
	os.makedirs(output_dir)

if args.sim and not os.path.exists(channel_status_file):
	sys.exit('No channel status map found for this simulation.\nPlease make one by running status_channel_sim.py')
#else:
#	print('\n\nERROR: reduce_data.py requires 3 arguments\n')
#	print('Usage:')
#	print('\tpython reduce_data.py <input_file> </path/to/output/directory/> </path/to/configuration/files/>')
#	sys.exit('\n')

print('\nReducing {}'.format( this_file ))
DataReduction.ReduceFile( this_file,\
                          output_dir,\
                          config_dir + '/Run_Parameters.csv',\
                          config_dir + '/Calibrations_Xe.csv',\
                          config_dir + '/Channel_Map.csv',\
                          fixed_trigger = True,\
                          fit_pulse_flag = False,\
                          num_events = -1,\
                          is_simulation = args.sim,\
                          save_hdf5=save_hdf5)

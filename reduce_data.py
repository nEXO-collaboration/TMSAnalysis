from TMSAnalysis.DataReduction import DataReduction
import sys
import os
import cProfile

this_file = sys.argv[1]
output_dir = sys.argv[2]
print('Reducing {}'.format(this_file))
#cProfile.run('DataReduction.ReduceH5File(this_file,output_dir,input_baseline=-1,input_baseline_rms=-1,polarity=1.,\
#				sampling_period=8.,fixed_trigger=True,trigger_position=1000,\
#				fit_pulse_flag=False)')
DataReduction.ReduceH5File(this_file,output_dir,\
				'./tms_analysis_config_files/Run_Parameters_Xe_Run29.csv',\
				'./tms_analysis_config_files/Calibrations_Xe_Run11b.csv',\
				'./tms_analysis_config_files/Channel_Map_Xe_Run29.csv',\
				input_baseline=-1,input_baseline_rms=-1,\
				fixed_trigger=True,fit_pulse_flag=False)

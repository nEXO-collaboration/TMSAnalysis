import pandas as pd
import numpy as np
import time
from TMSAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration

from TMSAnalysis.WaveformAnalysis import Waveform

#Classes for the Clustering Stage
from TMSAnalysis.Clustering import Signal
from TMSAnalysis.Clustering import SignalArray
from TMSAnalysis.Clustering import Clustering

##################################################################################################
def ReduceFile( filename, output_dir, run_parameters_file, calibrations_file, channel_map_file, \
                        num_events=-1, input_baseline=-1, input_baseline_rms=-1, \
                        fixed_trigger=False, fit_pulse_flag=False, is_simulation=False):

        filetitle = filename.split('/')[-1]
        filetitle_noext = filetitle.split('.')[0]
        outputfile = '{}_reduced.h5'.format(filetitle_noext)
        output_df_list = [] # For some reason, it's faster to concat a list of dataframes than
                            # to append dataframes one-by-one.

        analysis_config = StruckAnalysisConfiguration.StruckAnalysisConfiguration()
        analysis_config.GetRunParametersFromFile( run_parameters_file )
        analysis_config.GetCalibrationConstantsFromFile( calibrations_file )
        analysis_config.GetChannelMapFromFile( channel_map_file )

        start_time = time.time()
        try:
                # This block runs if the input file is in an HDF5 format. Othwerise, it
                # will raise an OSError
                input_df = pd.read_hdf(filename)
                n_ev = 0
                reduced_df = FillH5Reduced(filetitle, input_df, analysis_config, n_ev,\
                                        input_baseline, input_baseline_rms, fixed_trigger, fit_pulse_flag, num_events=-1)
                output_df_list.append(reduced_df)
        except OSError:
                # This block runs if the input file is not an HDF5 file (meaning it is
                # assumed to be a ROOT file).
                from TMSAnalysis.ParseStruck import NGMRootFile
                from TMSAnalysis.ParseSimulation import NEXOOfflineFile

                if is_simulation:
                   input_file = NEXOOfflineFile.NEXOOfflineFile( input_filename = filename,\
                                                  output_directory = output_dir,\
                                                  config = analysis_config,\
                                                  add_noise = False)
                else:
                   input_file = NGMRootFile.NGMRootFile( input_filename = filename,\
                                                  output_directory = output_dir,\
                                                  config = analysis_config)
                print('Channel map loaded:') 
                print(input_file.channel_map) 
                print('\n{} active channels.'.format(len(input_file.channel_map))) 
                n_entries = input_file.GetTotalEntries()
                n_channels = analysis_config.GetNumberOfChannels()
                n_events = n_entries/n_channels
                n_ev = 0
                while n_ev<n_events:
                        print('\tProcessing event {} at {:4.4}s...'.format(n_ev,time.time()-start_time))
                        if (n_ev+20)*n_channels < n_entries:
                             start_stop = [n_ev*n_channels,(n_ev+20)*n_channels]
                        else:
                             start_stop = [n_ev*n_channels,(n_events-1)*n_channels]

                        input_df = input_file.GroupEventsAndWriteToHDF5(save = False, start_stop=start_stop)  
                        
                        reduced_df = FillH5Reduced(filetitle, input_df, analysis_config, n_ev,\
                                                input_baseline, input_baseline_rms, fixed_trigger, fit_pulse_flag, num_events=-1)
                        output_df_list.append(reduced_df)
                        n_ev += 20

        output_df = pd.concat( output_df_list, axis=0, ignore_index=True, sort=False )
        output_df.to_hdf(output_dir + outputfile, key='df')     
        print('Run time: {:4.4}'.format(time.time()-start_time))


##################################################################################################
def FillH5Reduced(filetitle, input_df, analysis_config, event_counter,\
                input_baseline, input_baseline_rms, fixed_trigger, \
                fit_pulse_flag, num_events=-1):

        output_series = pd.Series()
        output_list = []
        input_columns = input_df.columns
        output_columns = [col for col in input_columns if (col!='Data') and (col!='Channels')]
        sampling_period_ns = 1./(analysis_config.run_parameters['Sampling Rate [MHz]']/1.e3)
        key_buffer = None
        for index, thisrow in input_df.iterrows():
                skip = False
                if (event_counter > num_events) and (num_events > 0):
                        break
                # Set all the values that are not the waveform/channel values
                for col in output_columns:
                        output_series[col] = thisrow[col]

                output_series['TotalTileEnergy'] = 0.
                output_series['TotalSiPMEnergy'] = 0.
                output_series['NumTileChannelsHit'] = 0
                output_series['NumXTileChannelsHit'] = 0
                output_series['NumYTileChannelsHit'] = 0
                output_series['NumSiPMChannelsHit'] = 0
                output_series['TimeOfMaxChannel'] = 0
                
                max_channel_val = 0.
                sig_array  = SignalArray.SignalArray()
                # Loop through channels, do the analysis, put this into the output series
                for ch_num in range(len(thisrow['Channels'])):
                        if skip:
                                continue
                        software_ch_num = thisrow['Channels'][ch_num]
                        #calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel( software_ch_num )
                        #decay_constant = analysis_config.GetDecayTimeForSoftwareChannel( software_ch_num )
                        polarity = -1.
                        if analysis_config.run_parameters['Sampling Rate [MHz]'] == 62.5:
                                polarity = 1.
                        w = Waveform.Waveform(input_data=thisrow['Data'][ch_num],\
                                                detector_type       = thisrow['ChannelTypes'][ch_num],\
                                                sampling_period_ns  = sampling_period_ns,\
                                                input_baseline      = input_baseline,\
                                                input_baseline_rms  = input_baseline_rms,\
                                                polarity            = polarity,\
                                                fixed_trigger       = fixed_trigger,\
                                                trigger_position    = analysis_config.run_parameters['Pretrigger Length [samples]'],\
                                                decay_time_us       = analysis_config.GetDecayTimeForSoftwareChannel( software_ch_num ),\
                                                calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel( software_ch_num ) )
                        try:
                                w.FindPulsesAndComputeAQs(fit_pulse_flag=fit_pulse_flag)
                        except IndexError:
                                print('Null waveform found in channel {}, event {} skipped'.format(ch_num,event_counter))
                                key_buffer = output_list[index-1].keys() # Grab keys from previous event
                                for key_buf in key_buffer:
                                        output_series[key_buf] = 0
                                skip = True
                                continue
                        for key in w.analysis_quantities.keys():
                                output_series['{} {} {}'.format(\
                                                                analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ),\
                                                                analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ),\
                                                                key)] = w.analysis_quantities[key]
                        if 'TileStrip' in analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ):
                                if w.analysis_quantities['Charge Energy'] > 5. * w.analysis_quantities['Baseline RMS']:
                                        output_series['NumTileChannelsHit'] += 1
                                        ch_pos = analysis_config.GetChannelPos(software_ch_num)
                                        
                                        signal = Signal.Signal(w.analysis_quantities['Charge Energy'], \
                                                        w.analysis_quantities['Drift Time'], \
                                                        software_ch_num, \
                                                        ch_pos,\
                                                        analysis_config.GetChannelNameForSoftwareChannel( software_ch_num )\
                                                        )
                                        sig_array.AddSignal(signal)

                                        if 'X' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ):
                                                output_series['NumXTileChannelsHit'] += 1
                                        if 'Y' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ):
                                                output_series['NumYTileChannelsHit'] += 1
                                        output_series['TotalTileEnergy'] += w.analysis_quantities['Charge Energy']
                                        if w.analysis_quantities['Charge Energy']**2 > max_channel_val**2:
                                                max_channel_val = w.analysis_quantities['Charge Energy']
                                                output_series['TimeOfMaxChannel'] = w.analysis_quantities['T90']
                        if 'SiPM' in analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ):
                                if w.analysis_quantities['Pulse Area'] > 0.:
                                        output_series['NumSiPMChannelsHit'] += 1
                                        output_series['TotalSiPMEnergy'] += w.analysis_quantities['Pulse Area']
                
                #First fill event level info
                output_series['WeightedPosX']     = sig_array.GetPos1D('X')
                output_series['WeightedPosY']     = sig_array.GetPos1D('Y')
                output_series['WeightedDriftTime']= sig_array.GetTime()
                output_series['WeightedPosZ']     = sig_array.GetTime()*analysis_config.GetDriftVelocity()

                output_series['Weighted Event Size Z'] = sig_array.GetTimeRMS()*analysis_config.GetDriftVelocity()
                output_series['Weighted Event Size X'] = sig_array.GetPosRMS('X')
                output_series['Weighted Event Size Y'] = sig_array.GetPosRMS('Y')

                #Now cluster the signals and save number of clusters
                cluster=Clustering.Clustering(sig_array)
                cluster.Cluster()
                output_series['NumberOfClusters'] = cluster.GetNumClusters()
                output_series['IsFull3D']         = cluster.Is3DEvent()
                output_series['Number3DClusters'] = cluster.GetNumber3D()

                output_series['Cluster Energies'] = [c.GetEnergy() for c in cluster.clusters]
                output_series['Cluster X-Pos'] = [c.GetPos1D('X') for c in cluster.clusters]
                output_series['Cluster Y-Pos'] = [c.GetPos1D('Y') for c in cluster.clusters]
                output_series['Cluster Drift Time'] = [c.GetTime() for c in cluster.clusters]
                output_series['Cluster Z-Pos'] = [c.GetTime()*analysis_config.GetDriftVelocity() for c in cluster.clusters]
                
        
                #print("E1: %.2f, E2: %.2f, X: %.2f, Y: %.2f, N: %i, N3D: %i, Is3D:%i"%(output_series['TotalTileEnergy'],
                #                                 sig_array.GetEnergy(),
                #                                 output_series['WeightedPosX'], 
                #                                 output_series['WeightedPosY'], 
                #                                 output_series['NumberOfClusters'],
                #                                 output_series['Number3DClusters'],
                #                                 output_series['IsFull3D']))

                # Append this event to the output dataframe
                output_series['File'] = filetitle
                output_series['Event'] = event_counter
                output_list.append(output_series)
                event_counter += 1
        output_df = pd.concat( output_list, axis=0, ignore_index=True, sort=False )
        return output_df



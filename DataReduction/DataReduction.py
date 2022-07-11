from StanfordTPCAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration
from StanfordTPCAnalysis.WaveformAnalysis import Waveform
import time, pickle, os
import pandas as pd
import numpy as np

#Classes for the Clustering Stage
from StanfordTPCAnalysis.Clustering import SignalArray
from StanfordTPCAnalysis.Clustering import Clustering
from StanfordTPCAnalysis.Clustering import Signal

##################################################################################################
def ReduceFile( filename, output_dir, run_parameters_file, calibrations_file, channel_map_file, \
                        num_events=-1, fixed_trigger=False, fit_pulse_flag=False, is_simulation=False, add_noise=False):

        filetitle = filename.split('/')[-1]
        dataset = filename.split('/')[-3]
        filetitle_noext = filetitle.split('.')[0]
        outputfile = '{}_reduced.h5'.format(filetitle_noext)
        output_df_list = [] # For some reason, it's faster to concat a list of dataframes than
                            # to append dataframes one-by-one.

        analysis_config = StruckAnalysisConfiguration.StruckAnalysisConfiguration()
        analysis_config.GetRunParametersFromFile( run_parameters_file, dataset )
        analysis_config.GetCalibrationConstantsFromFile( calibrations_file )
        analysis_config.GetChannelMapFromFile( channel_map_file, dataset )
        input_baseline = int(analysis_config.run_parameters['Baseline Length [samples]'])

        start_time = time.time()
        try:
                # This block runs if the input file is in an HDF5 format. Othwerise, it
                # will raise an OSError
                input_df = pd.read_hdf(filename)
                n_ev = 0
                reduced_df = FillH5Reduced(filetitle, input_df, analysis_config, n_ev,\
                                        input_baseline, fixed_trigger, fit_pulse_flag, num_events=-1)
                output_df_list.append(reduced_df)
        except OSError:
                # This block runs if the input file is not an HDF5 file (meaning it is
                # assumed to be a ROOT file).
                from StanfordTPCAnalysis.ParseStruck import NGMRootFile
                from StanfordTPCAnalysis.ParseSimulation import NEXOOfflineFile

                if is_simulation:

                   path = os.path.dirname(filename) 
                   pickled_fname = path + '/channel_status.p'
                   global ch_status_flag
                   ch_status_flag = os.path.exists(pickled_fname)
                   if ch_status_flag:
                       global ch_status
                       with open(pickled_fname,'rb') as f:
                           ch_status = pickle.load(f)
                   input_file = NEXOOfflineFile.NEXOOfflineFile( input_filename = filename,\
                                                  output_directory = output_dir,\
                                                  config = analysis_config,\
                                                  add_noise = add_noise, noise_lib_directory='/usr/workspace/nexo/jacopod/noise_double_ganged/')
                else:
                   input_file = NGMRootFile.NGMRootFile( input_filename = filename,\
                                                  output_directory = output_dir,\
                                                  config = analysis_config)

                print('Channel map loaded:') 
                print(input_file.channel_map) 
                print('\n{} active channels.'.format(len(input_file.channel_map)))
                n_entries = input_file.GetTotalEntries()
                n_channels = analysis_config.GetNumberOfChannels()
                n_events_in_file = n_entries if is_simulation else n_entries/n_channels
                n_events_to_process = num_events if (num_events < n_events_in_file and num_events>0) else n_events_in_file
                n_events_processed = 0

                while n_events_processed < n_events_to_process:
                        print('\tProcessing event {} at {:4.4}s...'.format(n_events_processed,time.time()-start_time))
                        start_stop = [n_events_processed,(n_events_processed+20)] if (n_events_processed+20 < n_events_to_process)\
                                else [n_events_processed,n_events_to_process]

                        if not is_simulation:
                             start_stop[0] = start_stop[0]*n_channels
                             start_stop[1] = start_stop[1]*n_channels

                        input_df = input_file.GroupEventsAndWriteToHDF5(save = False, start_stop=start_stop)
                        reduced_df = FillH5Reduced(filetitle, input_df, analysis_config, n_events_processed,\
                                                input_baseline, fixed_trigger,\
                                                fit_pulse_flag, is_simulation=is_simulation, num_events=-1)
                        output_df_list.append(reduced_df)
                        n_events_processed += 20

        output_df = pd.concat( output_df_list, axis=0, ignore_index=True, sort=False )
        output_df.to_hdf(output_dir + outputfile, key='df')     
        print('Run time: {:4.4}'.format(time.time()-start_time))


##################################################################################################
def FillH5Reduced(filetitle, input_df, analysis_config, event_counter,\
                input_baseline, fixed_trigger, \
                fit_pulse_flag, is_simulation=False, num_events=-1):

        output_series = pd.Series()
        output_df = pd.DataFrame()
        input_columns = input_df.columns
        output_columns = [col for col in input_columns if (col!='Data') and (col!='Channels')]
        sampling_period_ns = 1./(analysis_config.run_parameters['Sampling Rate [MHz]']/1.e3) if not is_simulation\
                             else 1./(analysis_config.run_parameters['Simulation Sampling Rate [MHz]']/1.e3)
        key_buffer = None
        row_counter  = 0
        for index, thisrow in input_df.iterrows():
                if any([ch not in ['SiPM','TileStrip'] for ch in thisrow['ChannelTypes']]):
                    print('Skipping Event %i'%event_counter)
                    event_counter += 1
                    row_counter += 1
                    continue
                skip = False
                if (event_counter > num_events) and (num_events > 0):
                        break
                # Set all the values that are not the waveform/channel values
                for col in output_columns:
                        output_series[col] = thisrow[col]

                output_series['TotalTileEnergy'] = 0.
                output_series['TotalSiPMEnergy'] = 0.
                output_series['TotalSiPMEnergyBPolar'] = 0.
                output_series['NumTileChannelsHit'] = 0
                output_series['NumXTileChannelsHit'] = 0
                output_series['NumYTileChannelsHit'] = 0
                output_series['NumSiPMChannelsHit'] = 0
                output_series['NumSiPMChannelsHitBPolar'] = 0
                output_series['TimeOfMaxChannel'] = 0
                output_series['LightSaturated'] = False
                
                max_channel_val = 0.
                sig_array  = SignalArray.SignalArray()
                summed_sipm_data = None

                # Loop through channels, do the analysis, put this into the output series
                for ch_num in range(len(thisrow['Channels'])):
                        if skip:
                                continue
                        software_ch_num = thisrow['Channels'][ch_num]
                        #calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel( software_ch_num )
                        #decay_constant = analysis_config.GetDecayTimeForSoftwareChannel( software_ch_num )
                        polarity = 1.

                        if is_simulation:
                             trigger_position = int( analysis_config.run_parameters['Pretrigger Length [samples]'] *\
                                                     analysis_config.run_parameters['Simulation Sampling Rate [MHz]'] /\
                                                     analysis_config.run_parameters['Sampling Rate [MHz]'] )
                             decay_time_us = 100000000.
                             calibration_constant = 1.
                        else:
                             trigger_position = analysis_config.run_parameters['Pretrigger Length [samples]']
                             decay_time_us = analysis_config.GetDecayTimeForSoftwareChannel( software_ch_num )
                             calibration_constant = analysis_config.GetCalibrationConstantForSoftwareChannel( software_ch_num )

                        wfm_data = thisrow['Data'][ch_num]
                        if is_simulation and ch_status_flag:
                             if analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ) in ch_status.keys():
                                  mean,sigma = ch_status[analysis_config.GetChannelNameForSoftwareChannel( software_ch_num )]
                                  wfm_data = np.random.normal(mean,sigma,len(wfm_data))
                        w = Waveform.Waveform(input_data=wfm_data,\
                                                detector_type       = thisrow['ChannelTypes'][ch_num],\
                                                sampling_period_ns  = sampling_period_ns,\
                                                input_baseline      = input_baseline,\
                                                polarity            = polarity,\
                                                fixed_trigger       = fixed_trigger,\
                                                trigger_position    = trigger_position,\
                                                decay_time_us       = decay_time_us,\
                                                calibration_constant = calibration_constant )
                        try:
                                w.FindPulsesAndComputeAQs(fit_pulse_flag=fit_pulse_flag)
                                #print(thisrow['ChannelTypes'][ch_num],analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ))
                                #print(thisrow['Channels'][ch_num],analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ))
                                #print(w.__dict__.keys())
                                if w.flag:
                                    output_series['LightSaturated'] = True
                        except IndexError:
                                print('Null waveform found in channel {}, event {} skipped'.format(ch_num,event_counter))
                                key_buffer = output_df[row_counter-1].keys() # Grab keys from previous event
                                for key_buf in key_buffer:
                                        output_series[key_buf] = 0
                                skip = True
                                continue
                        for key in w.analysis_quantities.keys():
                                output_series['{} {} {}'.format(\
                                                                analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ),\
                                                                analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ),\
                                                                key)] = w.analysis_quantities[key]
                        # Compute the combined quantities for the tile.
                        if 'TileStrip' in analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ):

                                if (w.analysis_quantities['Charge Energy'] > 5. * w.analysis_quantities['Baseline RMS']) and \
                                   (w.analysis_quantities['Charge Energy']>0.5):
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

                                if w.analysis_quantities['Pulse Height'] > 5.*w.analysis_quantities['Baseline RMS'] and abs(w.analysis_quantities['Pulse Time'] - trigger_position)<30:
                                        if w.TagLightPulse():
                                            output_series['NumSiPMChannelsHit'] += 1
                                            output_series['TotalSiPMEnergy'] += w.analysis_quantities['Pulse Area']
                                        output_series['NumSiPMChannelsHitBPolar'] += 1
                                        output_series['TotalSiPMEnergyBPolar'] += w.analysis_quantities['Pulse Area']
#                                if summed_sipm_data is None:
#                                        summed_sipm_data = w.corrected_data
#                                else:
#                                        summed_sipm_data += w.corrected_data

#                # Create a waveform object for the summed light signal. Should be already calibrated. 
#                if summed_sipm_data is not None:
#                    summed_sipm_wfm = Waveform.Waveform(input_data=summed_sipm_data,\
#                                                        detector_type       = 'SiPM',\
#                                                        sampling_period_ns  = sampling_period_ns,\
#                                                        input_baseline      = input_baseline,\
#                                                        polarity            = 1,\
#                                                        fixed_trigger       = fixed_trigger,\
#                                                        trigger_position    = trigger_position,\
#                                                        decay_time_us       = 1.e9,\
#                                                        calibration_constant = 1. )
#                    summed_sipm_wfm.FindPulsesAndComputeAQs(fit_pulse_flag=fit_pulse_flag)
#                    # Add AQ's from summed waveform to the output
#                    for key in summed_sipm_wfm.analysis_quantities.keys():
#                            output_series['Summed SiPM {}'.format(key)] = summed_sipm_wfm.analysis_quantities[key]
#                    output_series['Summed SiPM Waveform'] = summed_sipm_data 


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
                
                # Append this event to the output dataframe
                output_series['File'] = filetitle
                output_series['Event'] = event_counter
                output_df = output_df.append(output_series,ignore_index=True)
                event_counter += 1
                row_counter += 1
        return output_df


from StanfordTPCAnalysis.StruckAnalysisConfiguration import StruckAnalysisConfiguration
from StanfordTPCAnalysis.WaveformAnalysis import Waveform
import time, pickle, os
import pandas as pd
import numpy as np

#Classes for the Clustering Stage
from StanfordTPCAnalysis.Clustering import SignalArray
from StanfordTPCAnalysis.Clustering import Clustering
from StanfordTPCAnalysis.Clustering import Signal

from StanfordTPCAnalysis.ParseStruck import NGMRootFile
from StanfordTPCAnalysis.ParseStruck import NGMBinaryFile
from StanfordTPCAnalysis.ParseSimulation import NEXOOfflineFile

##################################################################################################
def ReduceFile( filename, output_dir, run_parameters_file, calibrations_file, channel_map_file, save_hdf5=False,\
                        num_events=-1, fixed_trigger=False, fit_pulse_flag=False, is_simulation=False):

        filetitle = filename.split('/')[-1]
        if is_simulation:
           dataset = 'simulations'
        else:
           dataset = filename.split('/')[-3]
        filetitle_noext = filetitle.split('.')[0]
        outputfile = '{}_reduced.h5'.format(filetitle_noext)
        output_df_list = [] # For some reason, it's faster to concat a list of dataframes than
                            # to append dataframes one-by-one.

        analysis_config = StruckAnalysisConfiguration.StruckAnalysisConfiguration()
        analysis_config.GetRunParametersFromFile( run_parameters_file, dataset )
        analysis_config.GetCalibrationConstantsFromFile( calibrations_file )
        analysis_config.GetChannelMapFromFile( channel_map_file, dataset )
        start_time = time.time()

        is_rootfile = False
        if is_simulation:
           path = os.path.dirname(filename) 
           pickled_fname = path + '/channel_status.p'
           global ch_status

           with open(pickled_fname,'rb') as f:
                ch_status = pickle.load(f)
           input_file = NEXOOfflineFile.NEXOOfflineFile( input_filename = filename,\
                                          output_directory = output_dir,\
                                          config = analysis_config,\
                                          add_noise = False, noise_lib_directory='/usr/workspace/nexo/jacopod/noise/')
        elif filename.endswith('.root'):
           input_file = NGMRootFile.NGMRootFile( input_filename = filename,\
                                          output_directory = output_dir,\
                                          config = analysis_config)
           is_rootfile = True
        elif filename.endswith('.bin'):
           input_file = NGMBinaryFile.NGMBinaryFile( input_filename = filename,\
                                                  output_directory = output_dir,\
                                                  config = analysis_config)
        else:
           extension = filename.split('.')[-1]
           print('\n********** ERROR **********')
           print('Unrecognized file extension: .{}\n\n'.format(extension))
           sys.exit(1)


        print('Channel map loaded:') 
        print(input_file.channel_map) 
        print('\n{} active channels.'.format(len(input_file.channel_map))) 
        n_entries = input_file.GetTotalEntries()
        print("n entries:", n_entries)
        n_channels = analysis_config.GetNumberOfChannels()
        #n_events_in_file = n_entries if is_simulation else n_entries/n_channels
        n_events_in_file = n_entries/n_channels if is_rootfile else n_entries
        n_events_to_process = num_events if (num_events < n_events_in_file and num_events>0) else n_events_in_file
        n_events_processed = 0
        loop_counter = 0

        while n_events_processed < n_events_to_process:
                print('\tProcessing event {}/{} at {:4.4}s...'.format(n_events_processed,\
                                                                      n_events_to_process,\
                                                                      time.time()-start_time))

                start_stop = [n_events_processed,(n_events_processed+100)] if (n_events_processed+100 < n_events_to_process)\
                        else [n_events_processed,n_events_to_process]

                if not is_simulation:
                     start_stop[0] = start_stop[0]*n_channels
                     start_stop[1] = start_stop[1]*n_channels
                print('Begin GroupEventsAndWrite...')
                input_df = input_file.GroupEventsAndWriteToHDF5(save = save_hdf5, start_stop=start_stop )
                print('Begin FillH5Reduced...')
                reduced_df = FillH5Reduced(filetitle, input_df, analysis_config, n_events_processed,\
                                        fit_pulse_flag, is_simulation=is_simulation, num_events=-1)
                output_df_list.append(reduced_df)
                n_events_processed += len(reduced_df)
                loop_counter += 1

        output_df = pd.concat( output_df_list, axis=0, ignore_index=True, sort=False )
        output_df.to_hdf(output_dir + outputfile, key='df')     
        print('Run time: {:4.4}'.format(time.time()-start_time))


##################################################################################################
def FillH5Reduced(filetitle, input_df, analysis_config, event_counter,\
                fit_pulse_flag, is_simulation=False, num_events=-1):

        NOISE_WFM_DEBUGGING = False
        NOISE_WFM_SUBTRACTION = False

        output_series = pd.Series()
        output_df = pd.DataFrame()
        input_columns = input_df.columns
        output_columns = [col for col in input_columns if (col!='Data') and (col!='Channels')]

        key_buffer = None
        row_counter  = 0
        start_time = time.time()
        print('Reducing {} events.'.format(len(input_df)))
        for index, thisrow in input_df.iterrows():
                #if the channel type is not yet built for this analysis. 
                if any([ch not in ['SiPM','TileStrip','Off'] for ch in thisrow['ChannelTypes']]):
                    print('Skipping Event %i'%event_counter)
                    event_counter += 1
                    row_counter += 1
                    continue
                #print every 25 events
                if row_counter % 25 == 0:
                   print('Processing event {}/{} at {:4.4} min'.format(row_counter, \
                                                                       len(input_df), \
                                                                       (time.time()-start_time)/60.))

                skip = False
                #a break statement if using a fixed number of events smaller than total events
                if (event_counter > num_events) and (num_events > 0):
                        break
                # Set all the values that are not the waveform/channel values
                for col in output_columns:
                        output_series[col] = thisrow[col]

                #global quantities, from data reduction
                output_series['TotalTileEnergy'] = 0.
                output_series['TotalSiPMEnergy'] = 0.
                output_series['NumTileChannelsHit'] = 0
                output_series['NumXTileChannelsHit'] = 0
                output_series['NumYTileChannelsHit'] = 0
                output_series['NumSiPMChannelsHit'] = 0
                output_series['TimeOfMaxChannel'] = 0
                output_series['LightSaturated'] = False
 
                
                sig_array  = SignalArray.SignalArray()

                # Objects to store charge tile data for noise subtraction
                charge_channel_dict = {} # dict will be indexed by integers
                hit_channel_positions_x = []
                hit_channel_positions_y = []

                # Loop through channels, do SiPM analysis, and store the charge channels in a dict.
                for ch_num in range(len(thisrow['Channels'])):
                        if skip:
                                continue

                        software_ch_num = thisrow['Channels'][ch_num]
                        wfm_data = thisrow['Data'][ch_num]

                        # Some channels in simulation are populated with nothing, and it 
                        # is easier to process them if they are a gaussian. This looks at 
                        # some pickle file to see which channels are like this.                         
                        if is_simulation:
                             if analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ) in ch_status.keys():
                                  mean,sigma = ch_status[analysis_config.GetChannelNameForSoftwareChannel( software_ch_num )]
                                  wfm_data = np.random.normal(mean,sigma,len(wfm_data))

                        w = Waveform.Waveform(analysis_config, input_data=wfm_data, sw_ch=software_ch_num,\
                                                detector_type = thisrow['ChannelTypes'][ch_num])
                        try:
                                w.FindPulsesAndComputeAQs()

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

                        # Store hit data and waveform for charge signals, for use in noise subtraction.
                        if 'TileStrip' in analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ):

                                charge_channel_dict[software_ch_num] = w 

                                if(w.analysis_quantities['Passes Threshold']):
                                     if 'X' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ):
                                         hit_channel_positions_x.append( analysis_config.GetChannelPos(software_ch_num)[0] )
     
                                     elif 'Y' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ):
                                         hit_channel_positions_y.append( analysis_config.GetChannelPos(software_ch_num)[1] )

                        # Compute the combined quantities for the SiPM array. 
                        if 'SiPM' in analysis_config.GetChannelTypeForSoftwareChannel( software_ch_num ):
                                if(w.analysis_quantities['Passes Threshold']):
                                        output_series['NumSiPMChannelsHit'] += 1
                                        output_series['TotalSiPMEnergy'] += w.analysis_quantities['Pulse Area']

########################################################################################

                # Create charge noise waveform.
                noise_data = None
                num_noise_strips = 0
                noise_channels_list = []
                if NOISE_WFM_DEBUGGING:
                    print('\n\n\n')
                    print('Hit X positions: {}'.format(hit_channel_positions_x))
                    print('Hit Y positions: {}'.format(hit_channel_positions_y))
                for software_ch_num, w in charge_channel_dict.items():

                    # If there's a signal, skip this one.
                    if (w.analysis_quantities['Passes Threshold']):
                          if NOISE_WFM_DEBUGGING: \
                             print('Too much energy on channel {}'.format(\
                                      analysis_config.GetChannelNameForSoftwareChannel( software_ch_num )))
                          continue
                    
                    # If the channel is within NOISE_DISTANCE of a hit channel, skip it.
                    NOISE_DISTANCE = 7. # units: mm
                    if 'X' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ) and \
                       any( [np.abs(analysis_config.GetChannelPos(software_ch_num)[0] - xhit) < NOISE_DISTANCE \
                                              for xhit in hit_channel_positions_x] ) :
                         if NOISE_WFM_DEBUGGING: 
                            print('{} too close to a hit in X'.format(\
                                         analysis_config.GetChannelNameForSoftwareChannel( software_ch_num )))
                         continue
                    if 'Y' in analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ) and \
                       any( [np.abs(analysis_config.GetChannelPos(software_ch_num)[1] - yhit) < NOISE_DISTANCE \
                                                     for yhit in hit_channel_positions_y] ) :
                         if NOISE_WFM_DEBUGGING: 
                            print('{} too close to a hit in Y'.format(\
                                          analysis_config.GetChannelNameForSoftwareChannel( software_ch_num )))
                         continue

                    # If the channel is one of the big ganged channels, skip it.
                    if analysis_config.GetNumDevicesInChannelForSoftwareChannel( software_ch_num ) > 2:
                         if NOISE_WFM_DEBUGGING: 
                            print('{} is a big ganged channel'.format(\
                                          analysis_config.GetChannelNameForSoftwareChannel( software_ch_num )))
                         continue

                    num_noise_strips += analysis_config.GetNumDevicesInChannelForSoftwareChannel( software_ch_num ) 
                    if noise_data is None:
                       noise_data = w.corrected_data
                    else:
                       noise_data += w.corrected_data
                    noise_channels_list.append(analysis_config.GetChannelNameForSoftwareChannel( software_ch_num ))
                
                output_series['Num Noise Strips'] = num_noise_strips

                # Divide the noise waveform by the *number of strips*, so it's per-strip.
                if noise_data is not None:
                   noise_data /= num_noise_strips
                output_series['Noise Waveform'] = noise_data

                # Run a second loop over the charge channels, to get noise-subtracted data.                
                for software_ch_num, w in charge_channel_dict.items():
                    if noise_data is not None and NOISE_WFM_SUBTRACTION:
                        w.data = w.data - noise_data * analysis_config.GetNumDevicesInChannelForSoftwareChannel( software_ch_num ) 

                        try:
                                w.FindPulsesAndComputeAQs(fit_pulse_flag=fit_pulse_flag)
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

                    if (w.analysis_quantities['Passes Threshold']):
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

                           #find maximum channel
                           max_channel_val = 0.
                           if w.analysis_quantities['Charge Energy'] > max_channel_val:
                                   max_channel_val = w.analysis_quantities['Charge Energy']
                                   output_series['TimeOfMaxChannel'] = w.analysis_quantities['T90']

                #First fill event level info
                output_series['WeightedPosX']     = sig_array.GetPos1D('X')
                output_series['WeightedPosY']     = sig_array.GetPos1D('Y')
                output_series['WeightedDriftTime']= sig_array.GetTime()
                output_series['WeightedPosZ']     = sig_array.GetTime()*analysis_config.GetDriftVelocity()

                output_series['Weighted Event Size Z'] = sig_array.GetTimeRMS()*analysis_config.GetDriftVelocity()
                output_series['Weighted Event Size X'] = sig_array.GetPosRMS('X')
                output_series['Weighted Event Size Y'] = sig_array.GetPosRMS('Y')

                if is_simulation:
                   output_series['MCElectrons'] = thisrow['MCElectrons']
                   output_series['MCPhotons'] = thisrow['MCPhotons']

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
                output_series['Noise channels list'] = noise_channels_list
        
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
                output_df = output_df.append(output_series,ignore_index=True)
                event_counter += 1
                row_counter += 1
        return output_df

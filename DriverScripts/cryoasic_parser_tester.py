from StanfordTPCAnalysis.ParseCryoAsic import CryoAsicFile, CryoAsicEventViewer

import sys
import os



infile = "../../CRYO ASIC/logbook/testdata/pulsedata.dat"
chmap = "../ParseCryoAsic/channel_map_template.txt"
tilemap = "../ParseCryoAsic/tile_map_template.txt"

#d = CryoAsicFile.CryoAsicFile(infile, chmap, tilemap)
#d.load_raw_data(100)
#d.group_into_pandas()
#d.save_to_hdf5()


view = CryoAsicEventViewer.CryoAsicEventViewer(infile[:-4] + ".h5")
view.plot_event_rawcryo(10)
#view.plot_event_xysep(4)
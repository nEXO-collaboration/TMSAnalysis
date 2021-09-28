from StanfordTPCAnalysis.ParseCryoAsic import CryoAsicFile

import sys
import os



infile = "../../CRYO ASIC/logbook/testdata/pulsedata.dat"
chmap = "../ParseCryoAsic/channel_map_template.txt"
tilemap = "../ParseCryoAsic/tile_map_template.txt"

d = CryoAsicFile.CryoAsicFile(infile, chmap, tilemap)
d.load_raw_data(100)
d.group_into_pandas()
d.save_to_hdf5()

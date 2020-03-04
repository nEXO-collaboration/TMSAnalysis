from TMSAnalysis.ParseStruck import NGMRootFile
import sys

input_file = sys.argv[1]
output_dir = sys.argv[2]

print('Input file: {}'.format(input_file))
print('Output directory {}'.format(output_dir))

file = NGMRootFile.NGMRootFile(input_filename=input_file,output_directory=output_dir,channel_map_file='channel_map_8ns_sampling_LONGTPC2.txt')
file.GroupEventsAndWriteToHDF5()

# TMSAnalysis

This is a python-based analysis package intended to process and
analyze data taken on the various detectors in the Gratta Lab at
Stanford. The input can be data files produced by either the NGMDaq
software package (used to read data from Struck 3316 digitizers)
or CAEN's Wavedump software. This code first converts the files into
a pandas-readable HDF5 format, then does the waveform processing to
produce reduced variables for analysis. The end result are HDF5 files
containing pandas dataframes.

### Prerequisites

Which python packages (and versions) you need to install:

* [pandas (0.24.1+)](https://pandas.pydata.org/docs/)
* [numpy (1.17.4+)](https://numpy.org/)
* [tables (3.6.1+)](https://www.pytables.org/)
* [uproot (latest)](https://github.com/scikit-hep/uproot)
* [scipy (1.2.1+)](https://www.scipy.org/install.html)
* [numba (latest)](http://numba.pydata.org/)

All of the above should be available via `pip install`. If you already have a
version of these, you can use `pip install --upgrade <package_name>`.

Note that if you're running on a cluster, you should create a virtual environment
where you can install software packages locally. This is done using `virtualenv`. If you
have questions about this, contact Brian.

### Installing

To install the software, clone it from this repository and put it somewhere on your machine.
Then run the setup script:
```
source /path/to/TMSAnalysis/setup.sh
```
This script just appends the parent directory for TMSAnalysis to the user's `PYTHONPATH`, so
you can import the TMSAnalysis module in your python scripts and/or Jupyter notebooks.


## Running the code on NGMDaq files

We work mostly with ROOT files produced by the NGMDaq software package (which was written
by Jason Newby of ORNL). These files are called `tier1` files.

To process `tier1` files, the code first converts the ROOT trees into pandas dataframes,
in which the waveforms from all channels are grouped together into events. This can be done
with the script `convert_data_to_hdf5.py` contained in the `TMSAnalysis/DriverScripts` directory,
by running:
```
python /path/to/TMSAnalysis/DriverScripts/convert_data_to_hdf5.py <input_file> </path/to/output/directory/>
```
where `<input_file>` is the absolute path to a `tier1` ROOT file, and the output directory is wherever
you want to write your data. This script also writes only 200 events per output file, so in general
you will end up with several HDF5 output files for each input ROOT file.

Once the new HDF5 files have been created, you can run the data reduction step using the
`reduce_data.py` script, via the command:
```
python /path/to/TMSAnalysis/DriverScripts/reduce_data.py <input_file> </path/to/output/directory/> </path/to/configuration/files/>
```
where `<input_file>` is now an HDF5 file that you made in the previous step. The new thing here is that
now you need some configuration files. These are:

* The Run Parameters table - see [Example](https://docs.google.com/spreadsheets/d/1_a5np_45Q3RD28KyxvfwPUAgzYLbc04wWJq26Fh22G4/edit?usp=sharing)
* The Calibrations table - see [Example](https://docs.google.com/spreadsheets/d/1rXRXEe0IBWPgIpwmnd8P4OAsJjiRXsxcnnTBvuM9l0Q/edit?usp=sharing)
* The Channel Map - see [Example](https://docs.google.com/spreadsheets/d/1kfQ1g7JiRv8LEUFZ-IhzWiNHxBoyt0SbndU7X9NW9io/edit?usp=sharing)

Each of these can be downloaded as a `.csv` file and placed in a common location, at which point that location can be
passed as an argument above. One will also need to edit the `reduce_data.py` script if the names of the files change.

We've included example configuration files in the repository, which can be found at `TMSAnalysis/config/`.

* NOTE: at present, the `convert_data_to_hdf5.py` script requires a channel map text file, which is
being deprecated in other parts of the code. I plan to fix this soon.
* NOTE: as of 7 April 2020, the smoothing window used to extract some charge parameters is hard-coded,
if it needs to be changed open  `WaveformAnalysis/Waveform.py` and edit the `ns_smoothing_window` variable


## Generating batch hdf5 databases

It is also possible to generate a reduced hdf5 file straight from the tier1 root files. This can be achieved in two ways:

* processing tier1 file one-by-one
* submit a batch job for the entire run

for the following scripts, all the generated files will be located in the target folder.

### One-by-one

by running:
```
python /path/to/TMSAnalysis/DriverScripts/reduce_data.py <input_file> </path/to/output_reduced_directory/> </path/to/configuration/files/>
```
refer to the file header for more details

### Batch submission
by running:
```
python /path/to/TMSAnalysis/DriverScripts/LLNLBatchDataReduction.py </path/to/input/tier1_directory/> </path/to/output_reduced_directory/> </path/to/configuration/files/>
```
a batch job will be submitted for each file in the folder. The submission options are stored in the `cmd_options` variable. A `.out`
file is written containing the stdout of the file with the same name of the tier1 root file. If this script is run more than once on
a specific folder, it will automatically skip all the alredy processed tier1 root files
Possible errors can be quickly checked by running.
```
python /path/to/TMSAnalysis/DriverScripts/check_batch_output_error.py </path/to/output_reduced_directory/>
```
if there is any error, this script will produce a `.log` file with in each line the name of the tier1 root file that failed,
along with the error, otherwise the the `.log` file will only contain the string `No errors in this folder`

## Grouping all the hdf5 file

In case all the reduced `.h5` files need to be stacked into one file it is possible to do so with the command
```
python /path/to/TMSAnalysis/DriverScripts/add_into_one_df.py </path/to/output_reduced_directory/>
```
This will write a file called `/path/to/output_reduced_directory/reduced_added.h5`. Having all the data into one dataframe should be faster to load
than dynamically load the different reduced files, in case an analysis of the entire run is required.

## Simulated File
It's possible to produce the reduced files also from tier1 simulated files. The steps are the same, except three main points one needs to be aware of:
* in the folder containing the raw simulated data (```/path/to/input/sim_tier1_directory/```), a file called ```channel_status.p``` needs to be present.
This file contains a dict with a list of the channels not recording events (the elements are mean and RMS of the baseline). To produce this file refer to
the header of ```/path/to/TMSAnalysis/DriverScripts/status_channel_sim.py```

* the flag ```--sim``` need to be parsed. For example, to run ```LLNLBatchDataReduction.py```, this is the command:
```
python /path/to/TMSAnalysis/DriverScripts/LLNLBatchDataReduction.py </path/to/input/tier1_directory/> </path/to/output_reduced_directory/> </path/to/configuration/files/> --sim
``` 
* in case noise needs to be added, make sure that the flag ```add_noise``` in ```DataReduction.py``` is True with the associated noise library and False otherwise

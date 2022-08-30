# StanfordTPCAnalysis

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
source /path/to/StanfordTPCAnalysis/setup.sh
```
This script just appends the parent directory for StanfordTPCAnalysis to the user's `PYTHONPATH`, so
you can import the StanfordTPCAnalysis module in your python scripts and/or Jupyter notebooks.


## Running the code on NGMDaq files

We work mostly with ROOT files produced by the NGMDaq software package (which was written
by Jason Newby of ORNL). These files are called `tier1` files.

To process `tier1` files, the code first converts the ROOT trees into pandas dataframes,
in which the waveforms from all channels are grouped together into events. Then these events are
processed using the `Waveform` class. All of this happens automatically in the `DataReduction` class,
and can be run using the `reduce_data.py` script, via the command:
```
python /path/to/StanfordTPCAnalysis/DriverScripts/reduce_data.py <input_file> </path/to/output/directory/> </path/to/configuration/files/>
```
where `<input_file>` is a `tier1` ROOT file.


If desired, one can save files at the intermediate step, where the waveforms have been grouped into events
but not yet processed. This can be done
with the script `convert_data_to_hdf5.py` contained in the `StanfordTPCAnalysis/DriverScripts` directory,
by running:
```
python /path/to/StanfordTPCAnalysis/DriverScripts/convert_data_to_hdf5.py <input_file> </path/to/output/directory/>
```
where `<input_file>` is the absolute path to a `tier1` ROOT file, and the output directory is wherever
you want to write your data. This script also writes only 200 events per output file, so in general
you will end up with several HDF5 output files for each input ROOT file. *NOTE: this script is no longer maintained
and may not work as advertised. The recommended way to access waveform-level information is using the `Event` class,
as illustrated in the tutorial notebook.*

You need three configuration files to run the data processing, which should be packaged with the software
in the `StanfordTPCAnalysis/config/<run_number>` directory. You can see examples below:

* The Run Parameters table - see [Example](https://docs.google.com/spreadsheets/d/1_a5np_45Q3RD28KyxvfwPUAgzYLbc04wWJq26Fh22G4/edit?usp=sharing)
* The Calibrations table - see [Example](https://docs.google.com/spreadsheets/d/1rXRXEe0IBWPgIpwmnd8P4OAsJjiRXsxcnnTBvuM9l0Q/edit?usp=sharing)
* The Channel Map - see [Example](https://docs.google.com/spreadsheets/d/1kfQ1g7JiRv8LEUFZ-IhzWiNHxBoyt0SbndU7X9NW9io/edit?usp=sharing)

The "calibrations" table will be in `.csv` format, while the other two will genearlly be in `.xlsx` format. 
The code should support configuration files in both `.xlsx` and `.csv` format, although the `.csv` format is no
longer supported and may give you some trouble. 


* NOTE: at present, the `convert_data_to_hdf5.py` script requires a channel map text file, which is
being deprecated in other parts of the code. I plan to fix this soon.
* NOTE: as of 7 April 2020, the smoothing window used to extract some charge parameters is hard-coded,
if it needs to be changed open  `WaveformAnalysis/Waveform.py` and edit the `ns_smoothing_window` variable
* NOTE: as 28 September 2020, the config files can be `.xlsx` as well. This solution was adopted because in the 30th liquefaction different dataset have slightly different run parameters and channel configuration. This information is stored in different sheets of the `.xlsx` file. In order to point to the correct configuration, the `tier1` files need to be stored in such a path:
```
path_to_dataset/dataset_name/raw_data/
```
where `dataset_name` has to name the sheet in the `.xlsx` file.


### Batch submission
Processing jobs can be submitted in batch mode to the LLNL computing queue by running:
```
python /path/to/StanfordTPCAnalysis/DriverScripts/LLNLBatchDataReduction.py </path/to/input/tier1_directory/> </path/to/output_reduced_directory/> </path/to/configuration/files/>
```
This script submits one batch job for each file in the folder. The submission options are stored in the `cmd_options` variable. A `.out`
file is written containing the stdout of the file with the same name of the tier1 root file. If this script is run more than once on
a specific folder, it will automatically skip all the already processed tier1 root files.

Possible errors can be quickly checked by running.
```
python /path/to/StanfordTPCAnalysis/DriverScripts/check_batch_output_error.py </path/to/output_reduced_directory/>
```
if there is any error, this script will produce a `.log` file with in each line the name of the tier1 root file that failed,
along with the error, otherwise the the `.log` file will only contain the string `No errors in this folder`

## Grouping all the hdf5 file

In case all the reduced `.h5` files need to be stacked into one file it is possible to do so with the command
```
python /path/to/StanfordTPCAnalysis/DriverScripts/add_into_one_df.py </path/to/output_reduced_directory/>
```
This will write a file called `/path/to/output_reduced_directory/reduced_added.h5`. Having all the data into one dataframe should be faster to load
than dynamically load the different reduced files, in case an analysis of the entire run is required.

## Simulated File
It's possible to produce the reduced files also from tier1 simulated files. The steps are the same, except three main points one needs to be aware of:
* in the folder containing the raw simulated data (```/path/to/input/sim_tier1_directory/```), a file called ```channel_status.p``` needs to be present.
This file contains a dict with a list of the channels not recording events (the elements are mean and RMS of the baseline). To produce this file refer to
the header of ```/path/to/StanfordTPCAnalysis/DriverScripts/status_channel_sim.py```

* the flag ```--sim``` need to be parsed. For example, to run ```LLNLBatchDataReduction.py```, this is the command:
```
python /path/to/StanfordTPCAnalysis/DriverScripts/LLNLBatchDataReduction.py </path/to/input/tier1_directory/> </path/to/output_reduced_directory/> </path/to/configuration/files/> --sim
``` 
* in case noise needs to be added, make sure that the flag ```add_noise``` in ```DataReduction.py``` is True with the associated noise library and False otherwise

## Prerequisite for processing the binary files from the Struck

The default output files from the digitizer is a `.bin` file coupled with a `-conf.root` file. A first conversion from `.bin` to `.root`, to start the processing pipeline. In order to do that, the NGMDaq software needs to be installed. Generally this step is not required as the data are converted to `.root`, after the data-taking. For more information on the installation, contact Jacopo.

### Binary/Root conversion

The it is possible to do a batch conversion of the binary files by running the script

### Hardcoded Variables
in ```DataReduction.py```:
* ```light_strip_thr```
* ```delay_samples```
* ```charge_min```
* ```NOISE_DISTANCE```
```
in ```Waveform.py```:
* ```light_pretrigger```
* ```light_area_end```
* ```light_area_length```
* ```ns_smoothing_window```
```
if Differentiator scheme is active:
* ```differentiator_window```
```


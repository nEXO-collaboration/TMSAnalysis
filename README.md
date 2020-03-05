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

What things you need to install the software and how to install them

* [pandas](https://pandas.pydata.org/docs/)
* [uproot](https://github.com/scikit-hep/uproot)
* [scipy](https://www.scipy.org/install.html)
* [numba](http://numba.pydata.org/)

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
with the script `convert_data_to_hdf5.py` contained in the `TMSAnalysis/scripts` directory, 
by running:
```
python /path/to/TMSAnalysis/scripts/convert_data_to_hdf5.py <input_file> </path/to/output/directory/>
``` 
where `<input_file>` is the absolute path to a `tier1` ROOT file, and the output directory is wherever 
you want to write your data. This script also writes only 200 events per output file, so in general
you will end up with several HDF5 output files for each input ROOT file. 

Once the new HDF5 files have been created, you can run the data reduction step using the 
`reduce_data.py` script, via the command:
```
python /path/to/TMSAnalysis/scripts/reduce_data.py <input_file> </path/to/output/directory/> </path/to/configuration/files/> 
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


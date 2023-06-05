#!/bin/bash

source $HOME/StanfordTPCAnalysis/setup.sh
python $HOME/StanfordTPCAnalysis/DriverScripts/reduce_data.py /p/lustre1/nexouser/data/StanfordData/vidal4/Run45/DS00_Rn220/bin/SIS3316Raw_20230521220959DS00_Run45_1.bin /p/lustre1/nexouser/data/StanfordData/vidal4/Run45/DS00_Rn220/bin/ /p/lustre1/nexouser/data/StanfordData/vidal4/Run45/config_4sig/

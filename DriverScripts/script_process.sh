#!/bin/bash

source $HOME/StanfordTPCAnalysis/setup.sh
python $HOME/StanfordTPCAnalysis/DriverScripts/reduce_data.py /p/lustre1/nexouser/data/StanfordData/vidal4/Run41/DS03_Rn220_purifier_fast/bin/SIS3316Raw_20230127094525DS03_Rn220_purfication_Run41_1.bin /p/lustre1/nexouser/data/StanfordData/vidal4/Run41/DS03_Rn220_purifier_fast/bin/ /p/lustre1/nexouser/data/StanfordData/vidal4/Run41/config/

directory=/p/lustre1/nexouser/data/StanfordData/vidal4/Run41/DS03_Rn220_purifier_fast/bin/
config_file=/p/lustre1/nexouser/data/StanfordData/vidal4/Run41/config/

for file in $directory; do
    file_out=`basename $file`
    file_out=$directory$file
    command="python $HOME/StanfordTPCAnalysis/DriverScripts/reduce_data.py $file $file_out $config_file"
    echo $command
    $command
done

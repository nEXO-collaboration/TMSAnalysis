#!/bin/bash

export CURRENTDIR=$(pwd)
export TMSCODEDIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $TMSCODEDIR
cd ../
export TMSPARENTDIR=$(pwd)
cd $CURRENTDIR

export PYTHONPATH=$TMSPARENTDIR:$PYTHONPATH

#dirname "${BASH_SOURCE[0]}"
#export PYTHONPATH="../":$PYTHONPATH

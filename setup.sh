#!/bin/bash

export CURRENTDIR=$(pwd)
export STANFORDTPCCODEDIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $STANFORDTPCCODEDIR
cd ../
export STANFORDTPCPARENTDIR=$(pwd)
cd $CURRENTDIR

export PYTHONPATH=$STANFORDTPCPARENTDIR:$PYTHONPATH

#dirname "${BASH_SOURCE[0]}"
#export PYTHONPATH="../":$PYTHONPATH

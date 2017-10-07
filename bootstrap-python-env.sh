#!/usr/bin/env bash
set -e

ENVPATH=$(dirname `realpath $0`)/mlcomp-python-env/
echo $ENVPATH

if [ -d $ENVPATH ]; then rm -rf $ENVPATH; fi

virtualenv -p python3 $ENVPATH
echo " --- virtualenv in $ENVPATH --- "
echo " --- activate virtualenv --- "

cd $(dirname `realpath $0`)/mlcomp/

source $ENVPATH/bin/activate
pip3 install -e .
deactivate
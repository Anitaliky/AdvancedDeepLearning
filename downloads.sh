#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate main
echo "start downloads"
python downloads.py
echo "done"
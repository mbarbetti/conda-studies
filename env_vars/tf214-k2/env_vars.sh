#!/bin/bash

IFS=':'
read -ra path_arr <<< "$PATH"
unset IFS

unset clean_path
start_insert=false

for path in "${path_arr[@]}";
do
  if [[ "$path" == *"$CONDA_PREFIX"* ]]; then 
    start_insert=true
  fi
  if [ "$start_insert" == true ]; then 
    clean_path="$clean_path:$path"
  fi
done

unset PATH
export PATH=${clean_path:1}

unset LD_LIBRARY_PATH
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

unset XLA_FLAGS

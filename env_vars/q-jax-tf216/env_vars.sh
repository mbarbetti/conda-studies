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
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/:${LD_LIBRARY_PATH}
NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))

for dir in $(ls -1d $NVIDIA_DIR/*/); do
    if [ -d "${dir}lib" ]; then
        export LD_LIBRARY_PATH="${dir}lib:$LD_LIBRARY_PATH"
        if [[ $(basename $dir) == 'cuda_nvcc' ]] ; then
            export PATH="${dir}bin:$PATH"
        fi
    fi
done

unset XLA_FLAGS
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CONDA_PREFIX}/lib

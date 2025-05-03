#!/bin/bash
#SBATCH --job-name=bale-tune --nodes=1 --ntasks=1 --mem-per-cpu=10G --cpus-per-task=4 --time=1-00:00:00 -p gpu --gres=gpu --mail-type=begin --mail-type=end --mail-type=fail
echo $CUDA_VISIBLE_DEVICES
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bale
export PYTHONPATH="${PYTHONPATH}:/PATH/TO/BALE"
echo "$1"
echo "$2"
echo "$3"
echo "$4"
echo "$5"
echo "$6"
python3 ./run.py "$1" "$2" "$3" "$4" "$5" "$6"
conda deactivate


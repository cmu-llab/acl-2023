set -e
set -u
set -o pipefail


exp_num=1
dataset=chinese_baxter

# set environment variables
export WORK_DIR=$(pwd)
export SRC_DIR=$WORK_DIR/src
export DATA_DIR=$WORK_DIR/data
export CONF_DIR=$WORK_DIR/conf

python $SRC_DIR/train.py --conf ${dataset}

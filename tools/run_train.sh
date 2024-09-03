#!/bin/bash

WORK_DIR="/data/home/user12/dl-projects/mmdetection" # 工程路径
JOB_NAME="apd_fpn_09031233" # 工作名
LOG_DIR="$WORK_DIR/logs/$JOB_NAME" # 日志文件夹
SCRIPT="$WORK_DIR/tools/train.py" # 脚本路径
ARGS="$WORK_DIR/configs/solov2/solov2_r50_fpn_1x_apd.py" # 参数
PYTHON_PATH="/data/home/user12/mambaforge/envs/openmmlab/bin/python"  # 解释器路径

# 创建日志目录
mkdir -p "$LOG_DIR"

# 使用 heredoc 提交作业
bsub << EOF
#BSUB -J $JOB_NAME
#BSUB -o $LOG_DIR/%J.out
#BSUB -e $LOG_DIR/%J.err
#BSUB -n 1
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process"

# 运行训练脚本
$PYTHON_PATH $SCRIPT $ARGS

EOF


#!/bin/bash

WORK_DIR="/data/home/user12/dl-projects/mmdetection" # 工程路径
JOB_NAME="wdd_yolox_tiny_10171942" # TODO:工作名(执行脚本/执行时间）
LOG_DIR="$WORK_DIR/logs/$JOB_NAME" # 日志文件夹
SCRIPT="$WORK_DIR/tools/train.py" # 脚本路径
ARGS="$WORK_DIR/configs/yolox/yolox_tiny_8xb8-300e_wdd.py" # TODO:参数（执行脚本文件名）
PYTHON_PATH="/data/home/user12/mambaforge/envs/openmmlab/bin/python"  # 解释器路径

# 监控脚本的路径
SCRIPT_DIR=$(dirname "$(realpath "$BASH_SOURCE")") # 获取当前脚本的目录
MONITOR_SCRIPT="$SCRIPT_DIR/monitor.sh" # TODO:监控脚本路径,与当前脚本在同一目录下

# 创建日志目录
mkdir -p "$LOG_DIR"

# 提交作业并获取作业ID
JOB_ID=$(bsub << EOF | grep -oE '[0-9]+'
#BSUB -J $JOB_NAME
#BSUB -o $LOG_DIR/%J.out
#BSUB -e $LOG_DIR/%J.err
#BSUB -n 1
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process"

# 运行训练脚本
cd $WORK_DIR
$PYTHON_PATH $SCRIPT $ARGS
EOF
)

echo "Job submitted with ID: $JOB_ID"

# 开启监控任务后台运行
nohup bash "$MONITOR_SCRIPT" "$JOB_ID" "$LOG_DIR" "from_main_script" > "$LOG_DIR/monitor.log" 2>&1 &
echo "Monitor script started in background. Check $LOG_DIR/monitor.log for details."

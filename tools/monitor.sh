#!/bin/bash

# 获取命令行传递的第一个参数作为 JOB_ID
JOB_ID=$1

# 检查是否提供了 JOB_ID
if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 missing JOB_ID."
  exit 1
fi

# 在这里继续使用 JOB_ID 进行其他操作
echo "Monitoring job with ID: $JOB_ID"

# 获取作业的详细信息
JOB_DETAILS=$(bjobs -noheader -o "exec_host job_name submit_time stat" "$JOB_ID")

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
    echo "Failed to retrieve job details for job ID $JOB_ID. Please check the JOB_ID and try again."
    exit 1
fi

# 检查是否找到了作业
if [ -z "$JOB_DETAILS" ]; then
    echo "Job ID $JOB_ID not found. Please check the JOB_ID and try again."
    exit 1
fi

# 提取详细信息
EXEC_HOST=$(echo "$JOB_DETAILS" | awk '{print $1}')
JOB_NAME=$(echo "$JOB_DETAILS" | awk '{print $2}')
SUBMIT_TIME=$(echo "$JOB_DETAILS" | awk '{print $3 " " $4}')

# 发送邮件提醒开始监控
echo -e "Your LSF job $JOB_ID has been monitored.\n\nJob Details:\nHost: $EXEC_HOST\nName: $JOB_NAME\nSubmit Time: $SUBMIT_TIME" | mail -s "LSF Job Start Notification" kmh1102@126.com

while true; do
    # 检查任务状态的逻辑
    JOB_STATUS=$(bjobs -noheader -o stat "$JOB_ID")

    if [ $? -ne 0 ]; then
        echo "Failed to retrieve job status for job ID $JOB_ID. Please check the JOB_ID and try again."
        exit 1
    fi

    if [ -z "$JOB_STATUS" ]; then
        echo "Job ID $JOB_ID not found. Please check the JOB_ID and try again."
        exit 1
    fi

    if [ "$JOB_STATUS" == "DONE" ] || [ "$JOB_STATUS" == "EXIT" ]; then
        # 发送邮件通知
        echo -e "Your LSF job $JOB_ID has finished with status $JOB_STATUS.\n\nJob Details:\nHost: $EXEC_HOST\nName: $JOB_NAME\nSubmit Time: $SUBMIT_TIME" | mail -s "LSF Job Stop Notification" kmh1102@126.com
        break
    fi

    # 等待一分钟后再检查
    sleep 30m
done & # 将整个脚本放入后台执行


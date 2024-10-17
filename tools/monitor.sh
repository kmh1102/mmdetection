#!/bin/bash

# 接收的邮箱
ACCEPT_EMAIL="kmh1102@126.com"

# 检查间隔
PRE_POLL_INTERVAL=(2 10 40)

# 轮询间隔
POLL_INTERVAL=30m

# 监控作业状态的函数
pre_monitor_job_status() {
    echo "The job "$job_id" pre-monitoring starts."
    for interval in "${PRE_POLL_INTERVAL[@]}"; do
        echo "Sleeping for $interval minutes..."
        sleep "${interval}m"

        job_status=$(bjobs -noheader -o "stat" "$job_id")
        echo "Job status: $job_status"

        if [ "$job_status" != "RUN" ]; then
            echo "Job status is not RUN. Exiting monitor script."
            exit 0
        fi
    done

    echo "The pre-monitoring is complete and the official monitoring starts."
}

# 检查参数是否提供
assert_arguments() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        if [ -z "$1" ];then
            echo "Usage: $0 missing job_id argument."
        else
            echo "Usage: $0 missing log_dir argument."
        fi
        exit 1
    else
        echo "Monitoring job with ID: $job_id"
        echo "Log directory: $log_dir"
    fi
}

# 获取作业详细信息
get_job_details() {
    local job_id=$1
    local details=$(bjobs -noheader -o "exec_host job_name submit_time stat" "$job_id")
    if [ $? -ne 0 ]; then
        echo "Failed to retrieve job details for job ID $job_id. Please check the JOB_ID and try again."
        exit 1
    elif [ -z "$details" ]; then
        echo "Job ID $job_id not found. Please check the JOB_ID and try again."
        exit 1
    fi
    echo "$details"
}

# 解析job_details
parse_job_details() {
    # 接收两个参数：job_id、job_details(from func:get_job_details)
    local job_id="$1"
    local job_details="$2"

    # 解析job_details,定义为全局变量
    exec_host=$(echo "$job_details" | awk '{print $1}')
    job_name=$(echo "$job_details" | awk '{print $2}')
    submit_time=$(echo "$job_details" | awk '{print $3 " " $4 " " $5}')  # OCT 17 12:52
    job_status=$(echo "$job_details" | awk '{print $6}')
}

# 发送邮件通知
send_mail() {
    local subject=$1
    local body=$2
    local attachment=$3

    if [ -n "$attachment" ] && [ -f "$attachment" ]; then
       echo -e "$body" | mail -s "$subject" -a "$attachment" "$ACCEPT_EMAIL" &
    else
       echo -e "$body" | mail -s "$subject" "$ACCEPT_EMAIL" &
    fi
}


# 生成开始训练的正文
send_start_mail() {
    # 使用Here Document构造正文
    local start_body=$(cat <<EOF
Your LSF job $job_id has been monitored.

Job Details:
Host: $exec_host
Name: $job_name
Submit Time: $submit_time
Status: $job_status

The monitoring script is running with process ID: $script_pid
EOF
    )

    send_mail "LSF Job Start Notification" "$start_body"
}

# 生成结束训练的正文
send_end_mail() {
    local body="Your LSF job $job_id has finished with status $job_status.\n\n"
    body+="Job Details:\n"
    body+="Host: $exec_host\n"
    body+="Name: $job_name\n"
    body+="Submit Time: $submit_time\n"

    if [ "$job_status" == "EXIT" ]; then
        local err_file="$log_dir/$job_id.err"
        if [ -f "$err_file" ]; then
            if [ -s "$err_file" ]; then
                body+="\nNote: There were errors during job execution. Please check the attached error file.\n"
                send_mail "LSF Job Stop Notification with Errors" "$body" "$err_file"
            else
                body+="\nNote: The error file $err_file is empty.\n"
                send_mail "LSF Job Stop Notification, but err_file is empty." "$body"
            fi
        else
            body+="\nNote: The error file $err_file does not exist.\n"
            send_mail "LSF Job Exit Notification" "$body"
        fi
    else
        local out_file="$log_dir/$job_id.out"
        if [ -f "$out_file" ]; then
            if [ -s "$out_file" ]; then
                body+="\nNote: The job completed successfully. Please check the attached log file.\n"
                send_mail "LSF Job End Notification with Log" "$body" "$out_file"
            else
                body+="\nNote: The log file $out_file is empty.\n"
                send_mail "LSF Job End Notification, but log file is empty." "$body"
            fi
        else
            body+="\nNote: The log file $out_file does not exist.\n"
            send_mail "LSF Job Done Notification" "$body"
        fi
    fi
}

# 监控作业状态
monitor_job_status() {
    while true; do
        job_status=$(bjobs -noheader -o "stat" "$job_id") # 更新job_status

        if [ "$job_status" == "DONE" ] || [ "$job_status" == "EXIT" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Job $job_id has finished with status $job_status."
            # 发送训练终止的邮件通知
            send_end_mail
            break
        elif [ "$job_status" == "RUN" ];then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Job $job_id is still running."
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Job $job_id is interrupted unexpectedly with status $job_status."
            send_mail "LSF Job Interrupt Notification" "Job $job_id is interrupted unexpectedly. The status is $job_id."
            break
        fi

        sleep $POLL_INTERVAL
    done

    echo "The monitoring task for job $job_id is complete."
}

# 函数：转换相对路径为绝对路径
to_absolute_path() {
    local relative_path="$1"
    echo "$(readlink -f "$relative_path")"
}

main() {
    job_id="$1"
    log_dir="$2"
    call_from="$3" # 判断monitor.sh是否由其他脚本启动，手动启动时无需输出该参数
    script_pid=$$ # monitor.sh的后台进程号

    # 检查输出参数
    assert_arguments "$job_id" "$log_dir"

    # 判断是否是从主脚本调用
    if [ "$call_from" == "from_main_script" ]; then
        echo "Monitor script called by main script."
        # 预监控，仅在监控脚本由主脚本自动调用时运行
        pre_monitor_job_status
    else
        log_dir=$(to_absolute_path "$log_dir")
        mkdir -p "$log_dir"
        echo "Monitor script started in background. Check $log_dir/monitor.log for details." # 终端提示脚本开始运行
        exec > "$log_dir/monitor.log" 2>&1 # 重定向脚本输出到日志文件
        echo "Monitor script run from command line."
    fi

    local job_details=$(get_job_details "$job_id")

    # 解析job_details
    parse_job_details "$job_id" "$job_details"

    # 发送训练开始的通知邮件
    send_start_mail

    # 开始监控作业，后台运行
    monitor_job_status &
}

# 将主脚本运行在后台
main "$@"



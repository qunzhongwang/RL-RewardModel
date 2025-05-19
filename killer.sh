#!/bin/bash

# 输入参数
pid=$1  # 已知任务的 PID，从命令行传入
cd /m2v_intern/wangqunzhong/research/ddpo-pytorch
next_task="source rl_auxscripts/rl_sh/"$2"gpu_rl.sh --entropy_loss=True --reward_cot=False"  # 下一个任务

# 检查 PID 是否存在
if ! ps -p $pid > /dev/null; then
    echo "Error: No process found with PID $pid"
    exit 1
fi

echo "Managing task with PID: $pid"

# 等待 3 小时
echo "Waiting for 3 hours before stopping the task... and start $next_task"
sleep 3s

# 杀掉指定的任务
echo "Stopping task with PID: $pid"
kill $pid

# 检查是否成功停止
if ps -p $pid > /dev/null; then
    echo "Task with PID $pid is still running, forcing it to stop..."
    kill -9 $pid  # 强制停止
else
    echo "Task with PID $pid stopped successfully."
fi

# 启动下一个任务
echo "Starting next task: $next_task"
$next_task
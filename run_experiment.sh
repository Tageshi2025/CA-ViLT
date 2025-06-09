#!/bin/bash
###
 # @Author: Tageshi fuzongjing@foxmail.com
 # @Date: 2025-06-09 14:22:16
 # @LastEditors: Tageshi fuzongjing@foxmail.com
 # @LastEditTime: 2025-06-09 14:22:16
 # @FilePath: /CA-ViLT/run_experiment.sh
### 

# 设置默认配置文件和模式
CONFIG_FILE="config.yaml"
MODE="train"

# 解析命令行参数
while getopts "c:m:" opt; do
  case ${opt} in
    c) CONFIG_FILE=$OPTARG ;;  # 配置文件
    m) MODE=$OPTARG ;;         # 运行模式 (train or evaluate)
    \?) echo "Usage: $0 [-c config_file] [-m train|evaluate]"
        exit 1 ;;
  esac
done

# 打印配置信息
echo "Using config file: $CONFIG_FILE"
echo "Running in $MODE mode"

# 设置日志路径和目录
LOG_DIR="./logs/$(date +'%Y-%m-%d_%H-%M-%S')"
mkdir -p $LOG_DIR
echo "Logs will be saved in $LOG_DIR"

# 创建日志文件
LOG_FILE="$LOG_DIR/experiment.log"

# 设置环境变量
export PYTHONPATH=$(pwd)

# 启动训练或评估
if [ "$MODE" == "train" ]; then
    echo "Starting training..."
    python train.py --config $CONFIG_FILE | tee $LOG_FILE
elif [ "$MODE" == "evaluate" ]; then
    echo "Starting evaluation..."
    python eval.py --config $CONFIG_FILE | tee $LOG_FILE
else
    echo "Invalid mode. Use 'train' or 'evaluate'."
    exit 1
fi

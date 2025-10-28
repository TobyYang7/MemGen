#!/bin/bash

# GPU进程清理脚本
# 用于清理GPU上的僵尸进程

echo "=========================================="
echo "GPU 进程清理脚本"
echo "=========================================="
echo ""

# 检查当前GPU状态
echo "1. 当前GPU状态："
nvidia-smi
echo ""

# 查找占用GPU的进程
echo "2. 查找占用GPU设备的进程..."
GPU_PIDS=$(fuser -v /dev/nvidia* 2>&1 | grep -v "Cannot stat" | grep python | awk '{print $2}' | sort -u)

if [ -z "$GPU_PIDS" ]; then
    echo "   未发现Python进程占用GPU"
else
    echo "   发现以下Python进程占用GPU："
    for pid in $GPU_PIDS; do
        # 获取进程详细信息
        if ps -p $pid > /dev/null 2>&1; then
            echo "   PID: $pid"
            ps aux | grep -E "^.*\s+${pid}\s+" | grep -v grep | head -1 | awk '{print "   命令: " substr($0, index($0,$11))}'
        fi
    done
    echo ""
    
    # 询问是否清理
    read -p "是否清理这些进程？(y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "3. 正在清理进程..."
        for pid in $GPU_PIDS; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "   终止进程 $pid"
                kill -9 $pid
            fi
        done
        echo "   清理完成！"
        echo ""
        
        # 等待一下让GPU释放资源
        sleep 2
        
        # 再次检查GPU状态
        echo "4. 清理后的GPU状态："
        nvidia-smi
    else
        echo "   取消清理操作"
    fi
fi

echo ""
echo "=========================================="
echo "脚本执行完毕"
echo "=========================================="


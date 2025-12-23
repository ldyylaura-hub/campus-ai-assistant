#!/bin/bash

echo "========================================================"
echo "正在启动 校园知识库助手 (Campus AI Assistant)..."
echo "========================================================"

# 1. 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装。"
    exit 1
fi

# 2. 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[1/3] 正在创建虚拟环境 (venv)..."
    python3 -m venv venv
fi

# 3. 激活并安装依赖
echo "[2/3] 正在检查并安装依赖..."
source venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 启动
echo "[3/3] 正在启动 Streamlit 应用..."
echo ""
streamlit run app.py

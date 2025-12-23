@echo off
chcp 65001
echo ========================================================
echo 正在启动 校园知识库助手 (Campus AI Assistant)...
echo ========================================================

:: 1. 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8 或更高版本。
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b
)

:: 2. 创建虚拟环境 (如果不存在)
if not exist "venv" (
    echo [1/3] 正在创建虚拟环境 (venv)...
    python -m venv venv
)

:: 3. 激活虚拟环境并安装依赖
echo [2/3] 正在检查并安装依赖...
call venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 4. 启动应用
echo [3/3] 正在启动 Streamlit 应用...
echo.
echo 请在浏览器中访问显示的 URL (通常是 http://localhost:8501)
echo.
streamlit run app.py

pause

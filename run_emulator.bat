@echo off
echo ============================================================
echo 🌐 RL-Cybersecurity: 模拟器模式
echo ============================================================
echo.
echo 此模式需要3个独立的终端窗口
echo.
echo 终端1: HTTP服务器（此窗口）
echo 终端2: 训练智能体
echo 终端3: 流量生成器
echo.
echo 在端口8080上启动HTTP服务器...
echo.

python emulator/server.py

echo.
echo 服务器已停止。按任意键退出...
pause >nul
